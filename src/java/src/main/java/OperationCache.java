import java.util.concurrent.ForkJoinPool;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.UnmanagedMemory;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.word.Pointer;
import org.graalvm.word.WordFactory;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

public class OperationCache {
    private static final OperationCache INSTANCE = new OperationCache();
    private final Cache<Long, OperationResult> cache;
    private final ForkJoinPool executorPool;

    private OperationCache() {

        this.executorPool = new ForkJoinPool(
            Runtime.getRuntime().availableProcessors()
        );

        this.cache = Caffeine.newBuilder()
                .maximumSize(1_048_576)
                .initialCapacity(262_144)
                .recordStats()
                .executor(executorPool)
                .build();
    }

    public static OperationCache getInstance() {
        return INSTANCE;
    }

    public void insert(long key, OperationResult result) {
        cache.put(key, result);
    }

    @CEntryPoint(name = "cacheInsert")
    public static void nativeInsert(IsolateThread thread, long key, double real, double imag, long uniqueTableKey) {
        OperationResult result = new OperationResult(real, imag, uniqueTableKey);
        getInstance().insert(key, result);
    }

    public OperationResult find(long key) {
        return cache.getIfPresent(key);
    }

    @CEntryPoint(name = "cacheFind")
    public static Pointer nativeFind(IsolateThread thread, long key) {
        OperationResult result = getInstance().find(key);
        if (result == null) {
            return WordFactory.nullPointer();
        }
        
        Pointer ptr = UnmanagedMemory.malloc(24);

        ptr.writeDouble(0, result.real());
        ptr.writeDouble(8, result.imag());
        ptr.writeLong(16, result.uniqueTableKey());
        return ptr;
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

    public static void main(String[] args) {
        System.out.println("OperationCache initialized");
    }
}