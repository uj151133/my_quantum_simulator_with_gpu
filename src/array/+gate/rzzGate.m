function Rzz = rzzGate(phi)
%RZZGATE この関数の概要をここに記述
%   詳細説明をここに記述
Rzz = [exp(-1i * phi / 2), 0, 0, 0; 0, exp(1i * phi / 2), 0, 0; 0, 0, exp(1i * phi / 2), 0; 0, 0, 0, exp(-1i * phi /2)];
end

