function BARENCO = barencoGate(alpha, phi, theta)
%BARENCOGATE この関数の概要をここに記述
%   詳細説明をここに記述
BARENCO = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, exp(1i * alpha) * cos(theta), -1i * exp(1i * (alpha - phi)) * sin(theta); 0, 0, -1i * exp(1i * (alpha + phi)) * sin(theta), exp(1i * alpha) * cos(theta)];
end

