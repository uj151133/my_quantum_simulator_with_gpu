#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

typedef struct {
    double real;
    double imag;
} Complex;

typedef struct {
    Complex matrix[2][2];
} QMDDGate;

id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;

void initializeMetal() {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
}

QMDDGate identityGate() {
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:sizeof(QMDDGate) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    id<MTLLibrary> library = [device newDefaultLibrary];
    id<MTLFunction> function = [library newFunctionWithName:@"identityGate"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:nil];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:resultBuffer offset:0 atIndex:0];
    
    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    QMDDGate *result = (QMDDGate *)[resultBuffer contents];
    return *result;
}

QMDDGate phaseGateWithDelta(double delta) {
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:sizeof(QMDDGate) options:MTLResourceStorageModeShared];
    id<MTLBuffer> deltaBuffer = [device newBufferWithBytes:&delta length:sizeof(double) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    id<MTLLibrary> library = [device newDefaultLibrary];
    id<MTLFunction> function = [library newFunctionWithName:@"phaseGate"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:nil];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:resultBuffer offset:0 atIndex:0];
    [encoder setBuffer:deltaBuffer offset:0 atIndex:1];
    
    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    QMDDGate *result = (QMDDGate *)[resultBuffer contents];
    return *result;
}