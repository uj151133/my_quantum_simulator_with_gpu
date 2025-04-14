function Rxx = rxxGate(phi)
%RXXGATE この関数の概要をここに記述
%   詳細説明をここに記述
Rxx = [cos(phi / 2), 0, 0, -1i * sin(phi / 2); 0, cos(phi / 2), -1i * sin(phi / 2), 0; 0, -1i * sin(phi / 2), cos(phi / 2), 0; -1i * sin(phi / 2), 0, 0, cos(phi / 2)];
end

