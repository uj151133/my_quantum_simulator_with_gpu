use dashmap::DashMap;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use num_complex::Complex64;
use std::ptr;
use std::collections::hash_map::DefaultHasher;
use std::sync::atomic::{AtomicBool, Ordering};
use lazy_static::lazy_static;

const TABLE_SIZE: i64 = 1_048_576;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CComplex {
    pub real: f64,
    pub imag: f64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CQMDDEdge {
    pub weight: CComplex,
    pub uniqueTableKey: u64,
    pub isTerminal: bool,
    pub depth: i32,
}

#[repr(C)]
pub struct CQMDDNode {
    pub edges_ptr: *mut CQMDDEdge,
    pub rows: u32,
    pub cols: u32,
}

#[derive(Debug, Clone)]
pub struct QMDDEdge {
    pub weight: Complex64,
    pub unique_table_key: u64,
    pub is_terminal: bool,
    pub depth: i32,
}

#[derive(Debug, Clone)]
pub struct QMDDNode {
    pub edges: Vec<Vec<QMDDEdge>>,
}

impl PartialEq for QMDDEdge {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f64 = 1e-10;
        
        self.unique_table_key == other.unique_table_key &&
        self.depth == other.depth &&
        self.is_terminal == other.is_terminal &&
        (self.weight.re - other.weight.re).abs() < EPSILON &&
        (self.weight.im - other.weight.im).abs() < EPSILON
    }
}

impl PartialEq for QMDDNode {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.edges == other.edges
    }
}

#[inline]
unsafe fn convert_from_ffi(c_node: &CQMDDNode) -> Result<QMDDNode, &'static str> {
    if c_node.edges_ptr.is_null() || c_node.rows == 0 || c_node.cols == 0 {
        return Err("invalid input");
    }
    
    let total_size = (c_node.rows * c_node.cols) as usize;
    let c_edges_slice = std::slice::from_raw_parts(c_node.edges_ptr, total_size);
    
    let mut rust_edges = Vec::with_capacity(c_node.rows as usize);
    
    for row in 0..c_node.rows {
        let mut row_edges = Vec::with_capacity(c_node.cols as usize);
        
        for col in 0..c_node.cols {
            let index = (row * c_node.cols + col) as usize;
            let c_edge = &c_edges_slice[index];
            
            let rust_edge = QMDDEdge {
                weight: Complex64::new(c_edge.weight.real, c_edge.weight.imag),
                unique_table_key: c_edge.uniqueTableKey,
                is_terminal: c_edge.isTerminal,
                depth: c_edge.depth,
            };
            
            row_edges.push(rust_edge);
        }
        
        rust_edges.push(row_edges);
    }
    
    Ok(QMDDNode { edges: rust_edges })
}

fn convert_to_ffi(rust_node: &QMDDNode) -> CQMDDNode {
    if rust_node.edges.is_empty() {
        return CQMDDNode {
            edges_ptr: ptr::null_mut(),
            rows: 0,
            cols: 0,
        };
    }
    
    let rows = rust_node.edges.len() as u32;
    let cols = rust_node.edges[0].len() as u32;
    let total_size = (rows * cols) as usize;
    
    let mut c_edges = Vec::with_capacity(total_size);
    
    for row in &rust_node.edges {
        for edge in row {
            c_edges.push(CQMDDEdge {
                weight: CComplex {
                    real: edge.weight.re,
                    imag: edge.weight.im,
                },
                uniqueTableKey: edge.unique_table_key as u64,
                isTerminal: edge.is_terminal,
                depth: edge.depth,
            });
        }
    }
    
    let edges_ptr = Box::into_raw(c_edges.into_boxed_slice()) as *mut CQMDDEdge;
    
    CQMDDNode {
        edges_ptr,
        rows,
        cols,
    }
}

lazy_static! {
    static ref UNIQUE_TABLE: DashMap<i64, Vec<Arc<QMDDNode>>> = DashMap::new();
}

#[no_mangle]
pub extern "C" fn init_unique_table() {
    init_shutdown_hook();
}

fn hash(id: u64) -> i64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    (hasher.finish() as i64) % TABLE_SIZE
}

#[no_mangle]
pub extern "C" fn find(unique_id: u64) -> *const CQMDDNode {
    let key = hash(unique_id);

    if let Some(bucket) = UNIQUE_TABLE.get(&key) {
        for existing_arc in bucket.iter() {
            // uniqueTableKeyで検索
            if !existing_arc.edges.is_empty() && 
               !existing_arc.edges[0].is_empty() &&
               existing_arc.edges[0][0].unique_table_key == unique_id {
                // Arc<QMDDNode>からCQMDDNodeを作成して返す
                // ここでは生ポインタを返すため、呼び出し側でメモリ管理が必要
                return Box::into_raw(Box::new(convert_to_ffi(&**existing_arc)));
            }
        }
    }
    
    ptr::null()
}

#[no_mangle]
pub extern "C" fn insert(unique_id: u64, node: *const CQMDDNode) -> bool {
    if node.is_null() {
        return false;
    }
    
    let c_node = unsafe { &*node };
    let rust_node = unsafe {
        match convert_from_ffi(c_node) {
            Ok(node) => node,
            Err(_) => return false,
        }
    };

    let key = hash(unique_id);
    let mut bucket = UNIQUE_TABLE.entry(key).or_insert_with(Vec::new);

    // 重複チェック
    for existing_arc in bucket.iter() {
        if **existing_arc == rust_node {
            return false;
        }
    }

    // 新規挿入
    bucket.push(Arc::new(rust_node));
    true
}

static SHUTDOWN_HOOK: std::sync::Once = std::sync::Once::new();
fn init_shutdown_hook() {
    SHUTDOWN_HOOK.call_once(|| {
        extern "C" fn cleanup_handler() {
            shutdown_unique_table();
        }
        
        unsafe {
            libc::atexit(cleanup_handler);
        }
    });
}

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
#[no_mangle]
pub extern "C" fn shutdown_unique_table() {
    if !SHUTDOWN.swap(true, Ordering::SeqCst) {
        UNIQUE_TABLE.clear();
        println!("UniqueTable cleared");
    }
}
