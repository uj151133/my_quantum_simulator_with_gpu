#ifndef QMDD_h
#define QMDD_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

@class QMDDNode;
@class QMDDEdge;
@class QMDDGate;
@class QMDDState;

typedef NS_ENUM(NSUInteger, OperationType) {
    OperationTypeAdd,
    OperationTypeMul,
    OperationTypeKronecker,
};

@interface QMDDEdge : NSObject

@property (nonatomic, assign) simd_double2 weight;
@property (nonatomic, assign) NSUInteger uniqueTableKey;
@property (nonatomic, assign) BOOL isTerminal;

- (instancetype)initWithWeight:(simd_double2)weight node:(QMDDNode *)node;
- (instancetype)initWithWeight:(double)weight node:(QMDDNode *)node;
- (instancetype)initWithWeight:(simd_double2)weight key:(NSUInteger)key;
- (instancetype)initWithWeight:(double)weight key:(NSUInteger)key;
- (QMDDNode *)getStartNode;
- (NSArray<NSValue *> *)getAllElementsForKet;
- (BOOL)isEqualToEdge:(QMDDEdge *)other;

@end

@interface QMDDNode : NSObject

@property (nonatomic, strong) NSArray<NSArray<QMDDEdge *> *> *edges;

- (instancetype)initWithEdges:(NSArray<NSArray<QMDDEdge *> *> *)edges;
- (BOOL)isEqualToNode:(QMDDNode *)other;

@end

@interface QMDDGate : NSObject

@property (nonatomic, strong) QMDDEdge *initialEdge;
@property (nonatomic, assign) NSUInteger depth;

- (instancetype)initWithEdge:(QMDDEdge *)edge numEdge:(NSUInteger)numEdge;
- (QMDDNode *)getStartNode;
- (QMDDEdge *)getInitialEdge;
- (NSUInteger)getDepth;
- (BOOL)isEqualToGate:(QMDDGate *)other;

@end

@interface QMDDState : NSObject

@property (nonatomic, strong) QMDDEdge *initialEdge;
@property (nonatomic, assign) NSUInteger depth;

- (instancetype)initWithEdge:(QMDDEdge *)edge;
- (QMDDNode *)getStartNode;
- (QMDDEdge *)getInitialEdge;
- (NSUInteger)getDepth;
- (NSArray<NSValue *> *)getAllElements;
- (QMDDState *)addState:(QMDDState *)other;
- (BOOL)isEqualToState:(QMDDState *)other;

@end

#endif