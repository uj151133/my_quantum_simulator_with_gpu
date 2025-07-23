using System.Collections.Generic;

namespace GUI.Shared.Models
{
    public class SimulationResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;
        public double ExecutionTime { get; set; }
        public string FinalState { get; set; } = string.Empty;
        public List<GateExecutionLog> GateExecutionLogs { get; set; } = new List<GateExecutionLog>();
    }

    public class GateExecutionLog
    {
        public int GateNumber { get; set; }
        public string GateLabel { get; set; } = string.Empty;
        public string GateType { get; set; } = string.Empty;
        public List<int> Qubits { get; set; } = new List<int>();
        public List<int>? ControlQubits { get; set; }
        public QMDDGateInfo CurrentGate { get; set; } = new QMDDGateInfo();
        public QMDDStateInfo CurrentState { get; set; } = new QMDDStateInfo();
    }

    public class QMDDGateInfo
    {
        public string Weight { get; set; } = string.Empty;
        public string Key { get; set; } = string.Empty;
        public int IsTerminal { get; set; }
    }

    public class QMDDStateInfo
    {
        public string Weight { get; set; } = string.Empty;
        public string Key { get; set; } = string.Empty;
        public int IsTerminal { get; set; }
    }
}
