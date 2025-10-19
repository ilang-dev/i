use libloading::{Library, Symbol};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use compiler::{
    backend::{c::CBackend, Build, Render},
    graph::Graph,
    l2::Lowerer,
    parser::Parser,
};

#[derive(Debug)]
#[repr(C)]
pub struct Tensor<'a> {
    pub data: *const f32,
    pub shape: *const usize,
    pub ndim: usize,
    pub _marker: std::marker::PhantomData<&'a [f32]>,
}

#[derive(Debug)]
#[repr(C)]
pub struct TensorMut<'a> {
    pub data: *mut f32,
    pub shape: *const usize,
    pub ndim: usize,
    pub _marker: std::marker::PhantomData<&'a mut [f32]>,
}

#[pyclass]
#[derive(Debug)]
struct Component {
    graph: Graph,
}

#[pymethods]
impl Component {
    #[new]
    fn new(src: String) -> PyResult<Self> {
        let (_ast, expr_bank) = Parser::new(&src).unwrap().parse().unwrap();
        let graph = Graph::from_expr_bank(&expr_bank);
        Ok(Component { graph })
    }

    fn __str__(&self) -> PyResult<String> {
        //Ok(format!("{:#?}", &self.graph))
        Ok(format!("{:#?}", Lowerer::new().lower(&self.graph)))
    }

    fn _code(&self) -> PyResult<String> {
        Ok(format!(
            "{}",
            &CBackend::render(&Lowerer::new().lower(&self.graph))
        ))
    }

    #[pyo3(name = "chain")]
    fn chain(&self, other: &Component) -> PyResult<Component> {
        Ok(Component {
            graph: self.graph.chain(&other.graph),
        })
    }

    #[pyo3(name = "__or__")]
    fn or(&self, other: &Component) -> PyResult<Component> {
        self.chain(other)
    }

    #[pyo3(name = "compose")]
    fn compose(&self, other: &Component) -> PyResult<Component> {
        Ok(Component {
            graph: self.graph.compose(&other.graph),
        })
    }

    #[pyo3(name = "__call__")]
    fn call(&self, other: &Component) -> PyResult<Component> {
        self.compose(other)
    }

    #[pyo3(name = "fanout")]
    fn fanout(&self, other: &Component) -> PyResult<Component> {
        Ok(Component {
            graph: self.graph.fanout(&other.graph),
        })
    }

    #[pyo3(name = "__and__")]
    fn and(&self, other: &Component) -> PyResult<Component> {
        self.fanout(other)
    }

    #[pyo3(name = "pair")]
    fn pair(&self, other: &Component) -> PyResult<Component> {
        Ok(Component {
            graph: self.graph.pair(&other.graph),
        })
    }

    #[pyo3(name = "__matmul__")]
    fn _pair(&self, other: &Component) -> PyResult<Component> {
        self.pair(other)
    }

    #[pyo3(signature = (*args))]
    fn exec(&self, args: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        let mut tensors: Vec<PyTensor> = args.extract()?;

        // convert to backend `Tensor`s
        let tensors = tensors
            .iter()
            .map(|tensor| Tensor {
                data: tensor.data.as_ptr(),
                shape: tensor.shape.as_ptr(),
                ndim: tensor.shape.len(),
                _marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        let block = Lowerer::new().lower(&self.graph);
        let dylib_path = CBackend::build(&CBackend::render(&block)).unwrap();
        let dylib_path = "lib.so";

        unsafe {
            let dylib = Library::new(&dylib_path).unwrap();
            let rank: Symbol<extern "C" fn() -> usize> = dylib.get(b"rank").unwrap();

            let fshape: Symbol<extern "C" fn(*const Tensor, usize, usize, *mut usize)> =
                dylib.get(b"shape").unwrap();
            let f: Symbol<unsafe extern "C" fn(*const Tensor, usize, *mut TensorMut)> =
                dylib.get(b"f").unwrap();

            let mut shape = vec![0; rank()];
            fshape(tensors.as_ptr(), tensors.len(), rank(), shape.as_mut_ptr());

            let mut data = vec![0f32; shape.iter().product()];
            let mut out = TensorMut {
                data: data.as_mut_ptr(),
                shape: shape.as_ptr(),
                ndim: shape.len(),
                _marker: std::marker::PhantomData,
            };

            f(tensors.as_ptr(), tensors.len(), &mut out);

            std::fs::remove_file(dylib_path).unwrap();

            Ok(PyTensor { data, shape })
        }
    }
}

#[pyclass(name = "Tensor")]
#[derive(Debug, FromPyObject)]
struct PyTensor {
    #[pyo3(get)]
    data: Vec<f32>,
    #[pyo3(get)]
    shape: Vec<usize>,
}

fn infer_shape(list: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
    let mut shape = Vec::new();
    let mut current = list.clone();

    loop {
        shape.push(current.len());

        if current.is_empty() {
            break;
        }

        let first_item = current.get_item(0)?;
        match first_item.downcast::<PyList>() {
            Ok(sublist) => current = sublist.clone(),
            Err(_) => break,
        }
    }

    Ok(shape)
}

fn validate_and_flatten(
    list: &Bound<'_, PyList>,
    shape: &[usize],
    dim: usize,
    data: &mut Vec<f32>,
) -> PyResult<()> {
    if dim >= shape.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Array has more dimensions than expected",
        ));
    }

    if list.len() != shape[dim] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Inconsistent shape: Expected {} elements at dimension {}, got {}",
            shape[dim],
            dim,
            list.len()
        )));
    }

    if dim == shape.len() - 1 {
        for element in list.iter() {
            let element = element.extract()?;
            data.push(element);
        }
    } else {
        for element in list.iter() {
            let sublist = element.downcast::<PyList>()?;
            validate_and_flatten(&sublist, shape, dim + 1, data)?;
        }
    }

    Ok(())
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(elements: &Bound<'_, PyList>) -> PyResult<Self> {
        let shape = infer_shape(elements)?;
        let mut data = Vec::new();
        validate_and_flatten(elements, &shape, 0, &mut data)?;

        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Data size {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            )));
        }

        Ok(Self { data, shape })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "Tensor(shape={:?}, data={:?})",
            self.shape, self.data
        ))
    }
}

#[pymodule]
fn ilang(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<Component>()?;
    Ok(())
}
