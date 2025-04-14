function fSim = fSimGate(theta, phi)
%FSIMGATE この関数の概要をここに記述
%   詳細説明をここに記述
fSim = [1, 0, 0, 0; 0, cos(theta), -1i * sin(theta), 0; 0, -1i * sin(theta), cos(theta), 0; 0, 0, 0, exp(1i * phi)];
end

