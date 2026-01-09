use libloading::{Library, Symbol};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyTuple};

use compiler::{
    backend::{c::CBackend, Build, Render},
    graph::Graph,
    l2::lower,
    parser::Parser,
};

#[derive(Debug)]
#[repr(C)]
pub struct Tensor<'a> {
    pub data: *const f32,
    pub shape: *const usize,
    pub rank: usize,
    pub _marker: std::marker::PhantomData<&'a [f32]>,
}

#[derive(Debug)]
#[repr(C)]
pub struct TensorMut<'a> {
    pub data: *mut f32,
    pub shape: *const usize,
    pub rank: usize,
    pub _marker: std::marker::PhantomData<&'a mut [f32]>,
}

#[pyclass]
#[derive(Debug)]
struct Component {
    graph: Graph,
}

impl Component {
    fn evaluate(&self, inputs: &[PyTensor]) -> PyResult<Vec<PyTensor>> {
        let tensors = inputs
            .iter()
            .map(|tensor| Tensor {
                data: tensor.data.as_ptr(),
                shape: tensor.shape.as_ptr(),
                rank: tensor.shape.len(),
                _marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        let block = lower(&self.graph);
        let dylib_path = CBackend::build(&CBackend::render(&block)).unwrap();

        let outputs: Vec<PyTensor> = unsafe {
            let dylib = Library::new(&dylib_path).unwrap();

            let count: Symbol<extern "C" fn() -> usize> = dylib.get(b"count").unwrap();
            let ranks_fn: Symbol<extern "C" fn(*mut usize)> = dylib.get(b"ranks").unwrap();
            let shapes_fn: Symbol<extern "C" fn(*const Tensor, *mut *mut usize)> =
                dylib.get(b"shapes").unwrap();
            let exec_fn: Symbol<extern "C" fn(*const Tensor, *mut TensorMut)> =
                dylib.get(b"exec").unwrap();

            let n_out = count();

            let mut ranks = vec![0usize; n_out];
            ranks_fn(ranks.as_mut_ptr());

            let mut out_shapes: Vec<Vec<usize>> = ranks.iter().map(|&r| vec![0usize; r]).collect();
            let mut shape_ptrs: Vec<*mut usize> =
                out_shapes.iter_mut().map(|v| v.as_mut_ptr()).collect();

            shapes_fn(tensors.as_ptr(), shape_ptrs.as_mut_ptr());

            let mut out_data: Vec<Vec<f32>> = out_shapes
                .iter()
                .map(|shape| {
                    let n_elem: usize = shape.iter().product();
                    vec![0f32; n_elem]
                })
                .collect();

            let mut outs: Vec<TensorMut> = (0..n_out)
                .map(|i| TensorMut {
                    data: out_data[i].as_mut_ptr(),
                    shape: out_shapes[i].as_ptr(),
                    rank: out_shapes[i].len(),
                    _marker: std::marker::PhantomData,
                })
                .collect();

            exec_fn(tensors.as_ptr(), outs.as_mut_ptr());

            std::fs::remove_file(&dylib_path).unwrap();

            (0..n_out)
                .map(|i| PyTensor {
                    data: std::mem::take(&mut out_data[i]),
                    shape: out_shapes[i].clone(),
                })
                .collect()
        };

        Ok(outputs)
    }
}

#[pymethods]
impl Component {
    #[new]
    fn new(src: String) -> PyResult<Self> {
        let expr = Parser::new(&src).unwrap().parse().unwrap();
        let graph = Graph::from_expr(&expr);
        Ok(Component { graph })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", lower(&self.graph)))
    }

    fn _code(&self) -> PyResult<String> {
        Ok(format!("{}", &CBackend::render(&lower(&self.graph))))
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
    fn exec<'py>(&self, py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Py<PyAny>> {
        let inputs: Vec<PyTensor> = args.extract()?;
        let outs = self.evaluate(&inputs)?;

        if outs.len() == 1 {
            let t = outs.into_iter().next().unwrap();
            let py_t = Py::new(py, t)?;
            let obj = py_t.into_py_any(py)?;
            Ok(obj)
        } else {
            let objs: Vec<Py<PyAny>> = outs
                .into_iter()
                .map(|t| {
                    let py_t = Py::new(py, t).unwrap();
                    py_t.into_py_any(py).unwrap()
                })
                .collect();

            let tuple = PyTuple::new(py, &objs)?;
            let obj = tuple.into_py_any(py)?;
            Ok(obj)
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
        match first_item.cast::<PyList>() {
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
            let sublist = element.cast::<PyList>()?;
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
