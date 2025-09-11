#![allow(non_snake_case)]

use dashmap::DashMap;
// use std::sync::Arc;
// use num_complex::Complex64;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use lazy_static::lazy_static;

const TABLE_SIZE: u64 = 1_048_576;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct CComplex {
    pub real: f64,
    pub imag: f64,
}

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct CQMDDEdge {
    pub weight: CComplex,
    pub uniqueTableKey: u64,
}

#[repr(C)]
pub struct CQMDDNode {
    pub edges_ptr: *mut CQMDDEdge,
    pub rows: u32,
    pub cols: u32,
}

// #[derive(Debug, Clone)]
// pub struct QMDDEdge {
//     pub weight: Complex64,
//     pub unique_table_key: u64,
// }

// #[derive(Debug, Clone)]
// pub struct QMDDNode {
//     pub edges: Vec<Vec<QMDDEdge>>,
// }

// impl PartialEq for QMDDEdge {
//     #[inline]
//     fn eq(&self, other: &Self) -> bool {
//         const EPSILON: f64 = 1e-10;
        
//         self.unique_table_key == other.unique_table_key &&
//         (self.weight.re - other.weight.re).abs() < EPSILON &&
//         (self.weight.im - other.weight.im).abs() < EPSILON
//     }
// }

// impl PartialEq for QMDDNode {
//     #[inline]
//     fn eq(&self, other: &Self) -> bool {
//         self.edges == other.edges
//     }
// }

// unsafe fn convert_from_FFI(c_node: &CQMDDNode) -> Result<Arc<QMDDNode>, &'static str> {
//     if c_node.edges_ptr.is_null() || c_node.rows == 0 || c_node.cols == 0 {
//         return Err("invalid input");
//     }
    
//     let total_size = (c_node.rows * c_node.cols) as usize;
//     let c_edges_slice = std::slice::from_raw_parts(c_node.edges_ptr, total_size);
    
//     let mut rust_edges = Vec::with_capacity(c_node.rows as usize);
    
//     for row in 0..c_node.rows {
//         let mut row_edges = Vec::with_capacity(c_node.cols as usize);
        
//         for col in 0..c_node.cols {
//             let index = (row * c_node.cols + col) as usize;
//             let c_edge = &c_edges_slice[index];
            
//             let rust_edge = QMDDEdge {
//                 weight: Complex64::new(c_edge.weight.real, c_edge.weight.imag),
//                 unique_table_key: c_edge.uniqueTableKey
//             };
            
//             row_edges.push(rust_edge);
//         }
        
//         rust_edges.push(row_edges);
//     }
    
//     Ok(Arc::new(QMDDNode { edges: rust_edges }))
// }

// fn convert_to_FFI(rust_node: &Arc<QMDDNode>) -> CQMDDNode {
//     if rust_node.edges.is_empty() {
//         return CQMDDNode {
//             edges_ptr: ptr::null_mut(),
//             rows: 0,
//             cols: 0,
//         };
//     }
    
//     let rows = rust_node.edges.len() as u32;
//     let cols = rust_node.edges[0].len() as u32;
//     let total_size = (rows * cols) as usize;
    
//     let mut c_edges = Vec::with_capacity(total_size);
    
//     for edge in rust_node.edges.iter().flatten() {
//         c_edges.push(CQMDDEdge {
//             weight: CComplex {
//                 real: edge.weight.re,
//                 imag: edge.weight.im,
//             },
//             uniqueTableKey: edge.unique_table_key as u64
//         });
//     }
    
//     let edges_ptr = Box::into_raw(c_edges.into_boxed_slice()) as *mut CQMDDEdge;
    
//     CQMDDNode {
//         edges_ptr,
//         rows,
//         cols,
//     }
// }

#[derive(Debug, Clone, Copy)]
struct NodePtr(*const CQMDDNode);

unsafe impl Send for NodePtr {}
unsafe impl Sync for NodePtr {}

impl NodePtr {
    fn new(ptr: *const CQMDDNode) -> Self {
        NodePtr(ptr)
    }
    
    fn as_ptr(self) -> *const CQMDDNode {
        self.0
    }
}

lazy_static! {
    static ref UNIQUE_TABLE: DashMap<u64, Vec<(u64, NodePtr)>> = DashMap::new();
}

#[no_mangle]
pub extern "C" fn init_unique_table() {
    init_shutdown_hook();
}

fn hash(id: u64) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    
    let mut hash = FNV_OFFSET_BASIS;
    let bytes = id.to_ne_bytes();
    
    for byte in bytes.iter() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash % TABLE_SIZE
}

#[no_mangle]
pub extern "C" fn find(unique_id: u64) -> *const CQMDDNode {
    // println!("find({}) called", unique_id);
    let key = hash(unique_id);

    if let Some(bucket) = UNIQUE_TABLE.get(&key) {
        for (stored_id, node_ptr) in bucket.iter() {
            if *stored_id == unique_id {
                // return Box::into_raw(Box::new(convert_to_FFI(existing_arc)));
                // println!("  MATCH FOUND! Returning ptr={:p}", node_ptr.as_ptr());
                return node_ptr.as_ptr();
            }
        }
    }
    println!("  Returning null");
    debug_print_table();
    ptr::null()
}

#[no_mangle]
pub extern "C" fn insert(unique_id: u64, node: *const CQMDDNode) -> bool {
    // println!("insert({}, {:p}) called", unique_id, node);
    if node.is_null() {
        return false;
    }
    
    // let c_node = unsafe { &*node };
    // let rust_node = unsafe {
    //     match convert_from_FFI(c_node) {
    //         Ok(node) => node,
    //         Err(_) => return false,
    //     }
    // };

    let key = hash(unique_id);
    let mut bucket = UNIQUE_TABLE.entry(key).or_insert_with(Vec::new);

    for (stored_id, _) in bucket.iter() {
        if *stored_id == unique_id {
            return false;
        }
    }

    // bucket.push((unique_id, rust_node));
    bucket.push((unique_id, NodePtr::new(node)));
    true
}

#[no_mangle]
pub extern "C" fn debug_print_table() {
    println!("=== UNIQUE_TABLE Contents ===");
    println!("Total buckets: {}", UNIQUE_TABLE.len());
    
    let mut total_entries = 0;
    let mut bucket_count = 0;
    
    for entry in UNIQUE_TABLE.iter() {
        let hash_key = entry.key();
        let bucket = entry.value();
        
        if !bucket.is_empty() {
            bucket_count += 1;
            println!("Bucket {} (hash key: {}): {} entries", bucket_count, hash_key, bucket.len());
            
            for (i, (unique_id, node_ptr)) in bucket.iter().enumerate() {
                total_entries += 1;
                println!("  Entry {}: unique_id={}, ptr={:p}", i + 1, unique_id, node_ptr.as_ptr());
                
                // ポインタの詳細情報も表示
                unsafe {
                    if !node_ptr.as_ptr().is_null() {
                        let node_ref = &*node_ptr.as_ptr();
                        println!("    Node details: rows={}, cols={}, edges_ptr={:p}", 
                               node_ref.rows, node_ref.cols, node_ref.edges_ptr);
                    } else {
                        println!("    Node details: NULL pointer");
                    }
                }
            }
        }
    }
    
    println!("Summary:");
    println!("  Non-empty buckets: {}", bucket_count);
    println!("  Total entries: {}", total_entries);
    println!("  Load factor: {:.6}", total_entries as f64 / TABLE_SIZE as f64);
    println!("=============================");
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
