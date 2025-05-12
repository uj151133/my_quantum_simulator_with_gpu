import com.github.benmanes.caffeine.cache.*;

public class OperationCache {
    private static final OperationCache INSTANCE = new OperationCache();
    private final Cache<Long, OperationResult> cache;

    private OperationCache() {
        this.cache = Caffeine.newBuilder()
                .maximumSize(1048576)
                .build();
    }

    public static OperationCache getInstance() {
        return INSTANCE;
    }

    public void insert(long key, OperationResult result) {
        cache.put(key, result);
    }

    public OperationResult find(long key) {
        return cache.getIfPresent(key);
    }

    public static native void nativeInsert(long key, double real, double imag, long size);
    public static native double[] nativeFind(long key);

    public static void doNativeInsert(long key, double real, double imag, long size) {
        OperationResult result = new OperationResult(real, imag, size);
        getInstance().insert(key, result);
    }

    public static OperationResult doNativeFind(long key) {
        return getInstance().find(key);
    }
}