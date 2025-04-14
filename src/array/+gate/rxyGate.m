function Rxy = rxyGate(phi)
%RXYGATE この関数の概要をここに記述
%   詳細説明をここに記述
Rxy = [1, 0, 0, 0; 0, cos(phi / 2), -1i * sin(phi /2), 0; 0, -1i * sin(phi / 2), cos(phi / 2), 0; 0, 0, 0, 1];
end

