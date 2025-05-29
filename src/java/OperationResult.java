public class OperationResult {
    public final double real;
    public final double imag;
    public final long uniqueTableKey;

    public OperationResult(double real, double imag, long uniqueTableKey) {
        this.real = real;
        this.imag = imag;
        this.uniqueTableKey = uniqueTableKey;
    }
}