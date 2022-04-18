function [T, relres] = evalAlignedUnwarp(A, imref)
%evalAlignedUnwarp - Description
%
% Syntax: relres = evalAlignedUnwarp(A, imref)
%
% Long description
    ta = 598400;
    x = rgb2gray(A);
    % x = A;
    y = rgb2gray(imref);
    s = sqrt(ta / size(y, 1) / size(y, 2));
    y = imresize(y, s);
    x = imresize(x, size(y));
    [vx, vy] = siftFlow(y, x);
    g = imgradient(y);
    g = g / max(g(:));
    % align
    [T, ~] = alignLD(g, vx, vy);
    [xx, yy] = meshgrid(1 : size(vx, 2), 1 : size(vy, 1));
    vx = T(1, 1) .* (xx + vx) + T(3, 1) - xx;
    vy = T(2, 2) .* (yy + vy) + T(3, 2) - yy;
    g = imresize(g, size(vx));
    vx = g .* vx;
    vy = g .* vy;
    t = sqrt(vx.^2 + vy.^2);
    relres = mean(t(:));
end