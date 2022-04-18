% add LD path
% https://people.csail.mit.edu/celiu/SIFTflow/
% change the path to your SIFTflow folder
addpath(genpath('~/SIFTflow'));
% GT images folder, for exampe: ~/data/docunet_benchmark/scan/
gtdir = '';
% Unwarped image folder
imdir = '';

tarea=598400;
% res = zeros(65, 4);
res = cell(65, 1);
% parpool(12);
parfor k = 1 : 65
    disp(k);
    rimg = imread(sprintf('%s/%d.png', gtdir, k));

    t = zeros(2, 5);
    for m = 1 : 2
        try
            ximg = imread(sprintf('%s/%d_%d copy.png', imdir, k, m));
            [ms, ld] = evalUnwarp(ximg, rimg);
            [~, relres] = evalAlignedUnwarp(ximg, rimg);
            t(m, :) = [k, m, relres, ms, ld];
        catch ME
            disp(ME.message)
            t(m, :) = [k, m, -1, -1, -1];
            % ms = 0;
            % ld = -1;
        end

    end
    res{k} = t;
end
res = cell2mat(res);
valres = res(res(:, 3) > 0, :);
avg = mean(valres, 1);
% avg = mean(res, 1);
res = cat(1, res, avg);

save(sprintf('%s/adres.txt', imdir), 'res', '-ascii');