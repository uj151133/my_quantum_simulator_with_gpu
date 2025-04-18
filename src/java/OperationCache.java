import com.github.benmanes.caffeine.cache.*;

public class OperationCache {
    private final Cache<Long, OperationResult> cache;

    public OperationCache() {
        cache = Caffeine.newBuilder()
            .maximumSize(10_000)
            .build();
    }

    public void put(long key, OperationResult result) {
        cache.put(key, result);
    }

    public OperationResult get(long key) {
        return cache.getIfPresent(key);
    }

    // JNIから呼ばれるメソッド（ネイティブに公開）
    public static native void putFromCpp(long key, double real, double imag, long index);
    public static native OperationResult getFromCpp(long key);
}
