function CZS = czsGate(theta, phi,gamma)
%CZSGATE この関数の概要をここに記述
%   詳細説明をここに記述
CZS = [1, 0, 0, 0; 0, -exp(1i * gamma) * sin(theta / 2 )^2 + cos(theta / 2)^2, 1/2 * (1 * exp(1i * gamma)) * exp(-1i * phi) * sin(theta), 0; 0, 1/2 * (1 + exp(1i * gamma)) * exp(1i * phi) * sin(theta), -exp(1i * gamma) * cos(theta / 2)^2 + sin(theta / 2)^2, 0; 0, 0, 0, -exp(1i * gamma)];
end

