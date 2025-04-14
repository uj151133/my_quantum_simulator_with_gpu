function SWAPAlpha = swapAlphaGate(alpha)
%SWAPALPHA この関数の概要をここに記述
%   詳細説明をここに記述
SWAPAlpha = [1, 0, 0, 0; 0, (1 + exp(1i * pi * alpha))/2, (1 - exp(1i * pi * alpha))/2, 0; 0, (1 - exp(1i * pi * alpha))/2, (1 + exp(1i * pi * alpha))/2, 0; 0, 0, 0, 1];
end

