// アイテムのリストを取得
const gates = [...document.querySelectorAll(".h-gate")];

const handleDragStart = (e) => e.target.classList.add("dragging");

const handleDragEnd = (e) => e.target.classList.remove("dragging");

for (const gate of gates) {
  gate.addEventListener("dragstart", handleDragStart, false);
  gate.addEventListener("dragend", handleDragEnd, false);
<<<<<<< HEAD
}
=======
}

window.dragAndDropInterop = {
    setData: function (e, format, data) {
        e.dataTransfer.setData(format, data);
    }
};
>>>>>>> c7167fbd64a790969422f8306258cadf1771fbef
