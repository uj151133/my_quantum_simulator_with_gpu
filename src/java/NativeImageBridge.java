import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

public class NativeImageBridge {
    
    @CEntryPoint(name = "operation_cache_insert")
    public static int insertOperation(IsolateThread thread, long key, double real, double imag, long uniqueTableKey) {
        try {
            OperationResult result = new OperationResult(real, imag, uniqueTableKey);
            OperationCache.getInstance().insert(key, result);
            return 0; // Success
        } catch (Exception e) {
            System.err.println("Error inserting operation: " + e.getMessage());
            return -1; // Error
        }
    }
    
    @CEntryPoint(name = "operation_cache_find")
    public static int findOperation(IsolateThread thread, long key, CCharPointer resultBuffer) {
        try {
            OperationResult result = OperationCache.getInstance().find(key);
            if (result == null) {
                return -1; // Not found
            }
            
            // 結果をバッファに書き込み (real, imag, uniqueTableKey の順)
            String resultJson = String.format("{\"real\":%.15f,\"imag\":%.15f,\"uniqueTableKey\":%d}", 
                result.real(), result.imag(), result.uniqueTableKey());
            
            try (CTypeConversion.CCharPointerHolder holder = CTypeConversion.toCString(resultJson)) {
                // バッファにコピー（実際の実装では適切なサイズチェックが必要）
                CCharPointer source = holder.get();
                for (int i = 0; source.read(i) != 0; i++) {
                    resultBuffer.write(i, source.read(i));
                }
                resultBuffer.write(resultJson.length(), (byte) 0); // null terminate
            }
            
            return 0; // Success
        } catch (Exception e) {
            System.err.println("Error finding operation: " + e.getMessage());
            return -2; // Error
        }
    }
    
    @CEntryPoint(name = "operation_cache_contains")
    public static int containsOperation(IsolateThread thread, long key) {
        try {
            return OperationCache.getInstance().contains(key) ? 1 : 0;
        } catch (Exception e) {
            System.err.println("Error checking operation: " + e.getMessage());
            return -1; // Error
        }
    }
    
    @CEntryPoint(name = "operation_cache_size")
    public static long getCacheSize(IsolateThread thread) {
        try {
            return OperationCache.getInstance().size();
        } catch (Exception e) {
            System.err.println("Error getting cache size: " + e.getMessage());
            return -1; // Error
        }
    }
    
    @CEntryPoint(name = "operation_cache_clear")
    public static int clearCache(IsolateThread thread) {
        try {
            OperationCache.getInstance().clear();
            return 0; // Success
        } catch (Exception e) {
            System.err.println("Error clearing cache: " + e.getMessage());
            return -1; // Error
        }
    }
}