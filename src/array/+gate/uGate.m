function U = uGate(theta, phi, lamda)
%UGATE この関数の概要をここに記述
%   詳細説明をここに記述
U = [cos(theta / 2), -exp(1i * lamda) * sin(theta / 2); exp(1i * phi) * sin(theta / 2), exp(1i * (lamda + phi)) * cos(theta / 2)];
end

