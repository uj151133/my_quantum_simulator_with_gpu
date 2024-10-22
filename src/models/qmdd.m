#import "QMDD.h"
#import "UniqueTable.h"
#import "Calculation.h"

@implementation QMDDEdge

- (instancetype)initWithWeight:(simd_double2)weight node:(QMDDNode *)node {
    self = [super init];
    if (self) {
        _weight = weight;
        _uniqueTableKey = node ? [Calculation generateUniqueTableKey:node] : 0;
        _isTerminal = !node;
        UniqueTable *table = [UniqueTable sharedInstance];
        if (![table find:_uniqueTableKey] && node) {
            [table insert:_uniqueTableKey node:node];
        }
    }
    return self;
}

- (instancetype)initWithWeight:(double)weight node:(QMDDNode *)node {
    return [self initWithWeight:simd_make_double2(weight, 0.0) node:node];
}

- (instancetype)initWithWeight:(simd_double2)weight key:(NSUInteger)key {
    self = [super init];
    if (self) {
        _weight = weight;
        _uniqueTableKey = key;
        _isTerminal = NO;
    }
    return self;
}

- (instancetype)initWithWeight:(double)weight key:(NSUInteger)key {
    return [self initWithWeight:simd_make_double2(weight, 0.0) key:key];
}

- (QMDDNode *)getStartNode {
    UniqueTable *table = [UniqueTable sharedInstance];
    return [table find:_uniqueTableKey];
}

- (NSArray<NSValue *> *)getAllElementsForKet {
    UniqueTable *table = [UniqueTable sharedInstance];
    QMDDNode *node = [table find:_uniqueTableKey];
    if (node.edges.count == 1) {
        @throw [NSException exceptionWithName:@"RuntimeException" reason:@"The start node has only one edge, which is not allowed." userInfo:nil];
    } else {
        NSMutableArray<NSValue *> *result = [NSMutableArray array];
        if (_isTerminal) {
            for (NSArray<QMDDEdge *> *edgeRow in node.edges) {
                for (QMDDEdge *edge in edgeRow) {
                    [result addObject:[NSValue valueWithBytes:&(simd_mul(_weight, edge.weight)) objCType:@encode(simd_double2)]];
                }
            }
        } else {
            NSArray<NSValue *> *child0 = [node.edges[0][0] getAllElementsForKet];
            NSArray<NSValue *> *child1 = [node.edges[1][0] getAllElementsForKet];
            [result addObjectsFromArray:child0];
            [result addObjectsFromArray:child1];
            for (NSUInteger i = 0; i < result.count; i++) {
                simd_double2 value;
                [result[i] getValue:&value];
                value = simd_mul(_weight, value);
                result[i] = [NSValue valueWithBytes:&value objCType:@encode(simd_double2)];
            }
        }
        return result;
    }
}

- (BOOL)isEqualToEdge:(QMDDEdge *)other {
    if (!other) {
        return NO;
    }
    if (!simd_equal(_weight, other.weight)) {
        return NO;
    }
    if (_isTerminal != other.isTerminal) {
        return NO;
    }
    if (!_isTerminal && _uniqueTableKey != other.uniqueTableKey) {
        return NO;
    }
    UniqueTable *table = [UniqueTable sharedInstance];
    if (!_isTerminal && [table find:_uniqueTableKey] != [table find:other.uniqueTableKey]) {
        return NO;
    }
    return YES;
}

@end

@implementation QMDDNode

- (instancetype)initWithEdges:(NSArray<NSArray<QMDDEdge *> *> *)edges {
    self = [super init];
    if (self) {
        _edges = edges;
    }
    return self;
}

- (BOOL)isEqualToNode:(QMDDNode *)other {
    if (!other) {
        return NO;
    }
    if (_edges.count != other.edges.count) {
        return NO;
    }
    for (NSUInteger i = 0; i < _edges.count; i++) {
        if (_edges[i].count != other.edges[i].count) {
            return NO;
        }
        for (NSUInteger j = 0; j < _edges[i].count; j++) {
            if (![_edges[i][j] isEqualToEdge:other.edges[i][j]]) {
                return NO;
            }
        }
    }
    return YES;
}

@end

@implementation QMDDGate

- (instancetype)initWithEdge:(QMDDEdge *)edge numEdge:(NSUInteger)numEdge {
    self = [super init];
    if (self) {
        _initialEdge = edge;
        _depth = 0;
        [self calculateDepth];
    }
    return self;
}

- (void)calculateDepth {
    UniqueTable *table = [UniqueTable sharedInstance];
    QMDDNode *currentNode = [table find:_initialEdge.uniqueTableKey];
    NSUInteger currentDepth = 0;

    while (currentNode && currentNode.edges.count > 0) {
        currentDepth++;
        currentNode = [table find:currentNode.edges[0][0].uniqueTableKey];
    }
    _depth = currentDepth;
}

- (QMDDNode *)getStartNode {
    UniqueTable *table = [UniqueTable sharedInstance];
    return [table find:_initialEdge.uniqueTableKey];
}

- (QMDDEdge *)getInitialEdge {
    return _initialEdge;
}

- (NSUInteger)getDepth {
    return _depth;
}

- (BOOL)isEqualToGate:(QMDDGate *)other {
    if (!other) {
        return NO;
    }
    if (![_initialEdge isEqualToEdge:other.initialEdge]) {
        return NO;
    }
    if (_depth != other.depth) {
        return NO;
    }
    return YES;
}

@end

@implementation QMDDState

- (instancetype)initWithEdge:(QMDDEdge *)edge {
    self = [super init];
    if (self) {
        _initialEdge = edge;
        _depth = 0;
        [self calculateDepth];
    }
    return self;
}

- (void)calculateDepth {
    UniqueTable *table = [UniqueTable sharedInstance];
    QMDDNode *currentNode = [table find:_initialEdge.uniqueTableKey];
    NSUInteger currentDepth = 0;

    while (currentNode && currentNode.edges.count > 0) {
        currentDepth++;
        currentNode = [table find:currentNode.edges[0][0].uniqueTableKey];
    }
    _depth = currentDepth;
}

- (QMDDNode *)getStartNode {
    UniqueTable *table = [UniqueTable sharedInstance];
    return [table find:_initialEdge.uniqueTableKey];
}

- (QMDDEdge *)getInitialEdge {
    return _initialEdge;
}

- (NSUInteger)getDepth {
    return _depth;
}

- (NSArray<NSValue *> *)getAllElements {
    return [_initialEdge getAllElementsForKet];
}

- (QMDDState *)addState:(QMDDState *)other {
    QMDDNode *newNode = [self addNodes:[self getStartNode] node2:[other getStartNode]];
    return [[QMDDState alloc] initWithEdge:[[QMDDEdge alloc] initWithWeight:simd_add(_initialEdge.weight, other.initialEdge.weight) node:newNode]];
}

- (QMDDNode *)addNodes:(QMDDNode *)node1 node2:(QMDDNode *)node2 {
    UniqueTable *table = [UniqueTable sharedInstance];
    if (!node1) {
        return node2;
    }
    if (!node2) {
        return node1;
    }

    NSMutableArray<NSMutableArray<QMDDEdge *> *> *resultEdges = [NSMutableArray arrayWithCapacity:node1.edges.count];
    for (NSUInteger i = 0; i < node1.edges.count; i++) {
        NSMutableArray<QMDDEdge *> *row = [NSMutableArray arrayWithCapacity:node1.edges[i].count];
        for (NSUInteger j = 0; j < node1.edges[i].count; j++) {
            QMDDEdge *edge1 = node1.edges[i][j];
            QMDDEdge *edge2 = node2.edges[i][j];
            QMDDEdge *newEdge = [[QMDDEdge alloc] initWithWeight:simd_add(edge1.weight, edge2.weight) node:[self addNodes:[table find:edge1.uniqueTableKey] node2:[table find:edge2.uniqueTableKey]]];
            [row addObject:newEdge];
        }
        [resultEdges addObject:row];
    }

    return [[QMDDNode alloc] initWithEdges:resultEdges];
}

- (BOOL)isEqualToState:(QMDDState *)other {
    if (!other) {
        return NO;
    }
    if (![_initialEdge isEqualToEdge:other.initialEdge]) {
        return NO;
    }
    return YES;
}

@end