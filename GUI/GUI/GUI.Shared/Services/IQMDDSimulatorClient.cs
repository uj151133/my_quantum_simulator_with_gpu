using GUI.Shared.Models;

namespace GUI.Shared.Services
{
    public interface IQMDDSimulatorClient
    {
        Task<bool> IsSimulatorAvailableAsync();
        Task<string> ExecuteQMDDAndParseOutput(string qasmInput);
        Task<SimulationResult> SimulateCircuitAsync(CircuitRequest request);
    }
}
