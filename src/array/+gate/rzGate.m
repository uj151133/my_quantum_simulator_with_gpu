function Rz = rzGate(theta)
%RZGATE この関数の概要をここに記述
%   詳細説明をここに記述
Rz = [exp(-1i * theta / 2), 0; 0, exp(1i * theta / 2)];
end

