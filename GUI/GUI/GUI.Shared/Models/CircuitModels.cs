using System.Collections.Generic;

namespace GUI.Shared.Models
{
    // JavaScriptから取得するデータ構造
    public class CircuitData
    {
        public List<QubitInfo> Qubits { get; set; } = new List<QubitInfo>();
        public List<GlobalGateInfo>? GlobalGateOrder { get; set; }
    }

    public class GlobalGateInfo
    {
        public int GateIndex { get; set; }
        public int QubitIndex { get; set; }
        public string Type { get; set; } = "";
        public long Timestamp { get; set; }
    }

    public class QubitInfo  
    {
        public int Index { get; set; }
        public List<GateInfo> Gates { get; set; } = new List<GateInfo>();
    }

    public class GateInfo
    {
        public string Type { get; set; } = "";
        public double Position { get; set; }
        public int? TargetQubit { get; set; }
        public double? Angle { get; set; }
    }

    // シミュレータに送信するリクエスト構造
    public class CircuitRequest
    {
        public int NumQubits { get; set; }
        public List<GateCommand> Gates { get; set; } = new List<GateCommand>();
    }

    public class GateCommand
    {
        public string Type { get; set; } = "";
        public List<int> Qubits { get; set; } = new List<int>();
        public List<int>? ControlQubits { get; set; }
        public double? Angle { get; set; }
    }
}
