use std::iter::Peekable;

use crate::block::{Arg, Block, Expr, Program, Statement, Type};

#[derive(Debug)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

pub fn parse(input: &str) -> Program {
    let tokens = tokenize(input).collect::<Vec<_>>();
    let mut iter = tokens.into_iter().peekable();
    let sexp = parse_sexp(&mut iter);
    let mut block = parse_block_sexp(&sexp);
    // TODO This could check the `Statement`s are of variant `Function`...
    let rank = block.statements.remove(0);
    let shape = block.statements.remove(0);
    let exec = block.statements.pop().unwrap();
    Program {
        library: Block {
            statements: block.statements,
        },
        rank,
        shape,
        exec,
    }
}

fn parse_block_sexp(sexp: &Sexp) -> Block {
    match sexp {
        Sexp::List(items) => {
            let mut statements = Vec::new();
            for item in items {
                statements.push(parse_statement(item));
            }
            Block { statements }
        }
        _ => Block::default(),
    }
}

fn parse_statement(sexp: &Sexp) -> Statement {
    match sexp {
        Sexp::List(list) if !list.is_empty() => {
            let head = &list[0];
            if let Sexp::Atom(keyword) = head {
                match keyword.as_str() {
                    "assign" => Statement::Assignment {
                        left: parse_expr(&list[1]),
                        right: parse_expr(&list[2]),
                    },
                    "decl" => Statement::Declaration {
                        ident: parse_atom(&list[1]),
                        type_: parse_type(&list[2]),
                        value: parse_expr(&list[3]),
                    },
                    "skip" => Statement::Skip {
                        index: parse_atom(&list[1]),
                        bound: parse_atom(&list[2]),
                    },
                    "loop" => Statement::Loop {
                        index: parse_atom(&list[1]),
                        bound: parse_expr(&list[2]),
                        parallel: parse_atom(&list[3]) == "1",
                        body: parse_block_sexp(&list[4]),
                    },
                    "return" => Statement::Return {
                        value: parse_expr(&list[1]),
                    },
                    "func" => Statement::Function {
                        ident: parse_atom(&list[1]),
                        args: parse_args(&list[2]),
                        body: parse_block_sexp(&list[3]),
                    },
                    "call" => Statement::Call {
                        ident: parse_atom(&list[1]),
                        args: parse_args(&list[2]),
                    },
                    _ => Statement::Skip {
                        index: "0".into(),
                        bound: "0".into(),
                    },
                }
            } else {
                Statement::Skip {
                    index: "0".into(),
                    bound: "0".into(),
                }
            }
        }
        _ => Statement::Skip {
            index: "0".into(),
            bound: "0".into(),
        },
    }
}

fn parse_args(sexp: &Sexp) -> Vec<Arg> {
    match sexp {
        Sexp::List(items) => items.iter().map(parse_arg).collect(),
        _ => vec![],
    }
}

fn parse_arg(sexp: &Sexp) -> Arg {
    match sexp {
        Sexp::List(list) if !list.is_empty() => {
            if let Sexp::Atom(a) = &list[0] {
                if a == "arg" {
                    return Arg {
                        type_: parse_type(&list[1]),
                        ident: parse_expr(&list[2]),
                    };
                }
            }
        }
        _ => {}
    }
    Arg {
        type_: Type::Int(false),
        ident: Expr::Ident("".into()),
    }
}

fn parse_expr(sexp: &Sexp) -> Expr {
    match sexp {
        Sexp::List(list) if !list.is_empty() => {
            if let Sexp::Atom(a) = &list[0] {
                match a.as_str() {
                    "alloc" => {
                        let val = parse_float(&list[1]);
                        let dims = parse_list_of_atoms(&list[2]);
                        Expr::Alloc {
                            initial_value: Box::new(Expr::Scalar(val)),
                            shape: dims,
                        }
                    }
                    "int" => {
                        let x = parse_atom(&list[1]).parse::<usize>().unwrap_or(0);
                        Expr::Int(x)
                    }
                    "scal" => {
                        let x = parse_atom(&list[1]).parse::<f32>().unwrap_or(0.);
                        Expr::Scalar(x)
                    }
                    "id" => Expr::Ident(parse_atom(&list[1])),
                    "ref" => Expr::Ref(parse_atom(&list[1]), false),
                    "ref!" => Expr::Ref(parse_atom(&list[1]), true),
                    "op" => {
                        let op_atom = parse_atom(&list[1]);
                        let mut inputs = Vec::new();
                        for i in &list[2..] {
                            inputs.push(parse_expr(i));
                        }
                        let op_char = op_atom.chars().next().unwrap_or('+');
                        Expr::Op {
                            op: op_char,
                            inputs,
                        }
                    }
                    "index" => Expr::Indexed {
                        ident: parse_atom(&list[1]),
                        index: Box::new(parse_expr(&list[2])),
                    },
                    _ => Expr::Ident("".into()),
                }
            } else {
                Expr::Ident("".into())
            }
        }
        _ => Expr::Ident("".into()),
    }
}

fn parse_type(sexp: &Sexp) -> Type {
    if let Sexp::Atom(a) = sexp {
        match a.as_str() {
            "i" => Type::Int(false),
            "i!" => Type::Int(true),
            "a" => Type::Array(false),
            "a!" => Type::Array(true),
            "ar" => Type::ArrayRef(false),
            "ar!" => Type::ArrayRef(true),
            _ => Type::Int(false),
        }
    } else {
        Type::Int(false)
    }
}

fn parse_atom(sexp: &Sexp) -> String {
    match sexp {
        Sexp::Atom(s) => s.clone(),
        _ => "".into(),
    }
}

fn parse_list_of_atoms(sexp: &Sexp) -> Vec<String> {
    if let Sexp::Atom(s) = sexp {
        s.split_whitespace().map(String::from).collect()
    } else {
        vec![]
    }
}

fn parse_float(sexp: &Sexp) -> f32 {
    parse_atom(sexp).parse::<f32>().unwrap_or(0.0)
}

fn parse_sexp<I>(iter: &mut Peekable<I>) -> Sexp
where
    I: Iterator<Item = Token>,
{
    match iter.peek() {
        Some(Token::OpenParen) => {
            iter.next();
            let mut items = Vec::new();
            while let Some(tok) = iter.peek() {
                if *tok == Token::CloseParen {
                    iter.next();
                    break;
                }
                items.push(parse_sexp(iter));
            }
            Sexp::List(items)
        }
        Some(Token::Atom(a)) => {
            let val = a.clone();
            iter.next();
            Sexp::Atom(val)
        }
        _ => Sexp::Atom("".into()),
    }
}

#[derive(Debug, PartialEq)]
enum Token {
    OpenParen,
    CloseParen,
    Atom(String),
}

fn tokenize(input: &str) -> impl Iterator<Item = Token> + '_ {
    let mut chars = input.chars().peekable();
    std::iter::from_fn(move || {
        skip_whitespace(&mut chars);
        match chars.peek().copied() {
            Some('(') => {
                chars.next();
                Some(Token::OpenParen)
            }
            Some(')') => {
                chars.next();
                Some(Token::CloseParen)
            }
            Some(_) => Some(Token::Atom(read_atom(&mut chars))),
            None => None,
        }
    })
}

fn skip_whitespace(chars: &mut Peekable<impl Iterator<Item = char>>) {
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
}

fn read_atom(chars: &mut Peekable<impl Iterator<Item = char>>) -> String {
    let mut s = String::new();
    while let Some(&c) = chars.peek() {
        if c == '(' || c == ')' || c.is_whitespace() {
            break;
        }
        s.push(c);
        chars.next();
    }
    s
}
