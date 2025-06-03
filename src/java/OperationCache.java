import com.github.benmanes.caffeine.cache.*;

public class OperationCache {
    private static final OperationCache INSTANCE = new OperationCache();
    private final Cache<Long, OperationResult> cache;

    private OperationCache() {
        this.cache = Caffeine.newBuilder()
                .maximumSize(1_048_576)
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

    public boolean contains(long key) {
        return cache.getIfPresent(key) != null;
    }
    
    public void invalidate(long key) {
        cache.invalidate(key);
    }
    
    public void clear() {
        cache.invalidateAll();
    }
    
    public long size() {
        return cache.estimatedSize();
    }

    public void printStats() {
        System.out.println("Cache size: " + cache.estimatedSize());
    }
}