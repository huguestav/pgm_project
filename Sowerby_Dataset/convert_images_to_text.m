a = load('sowerby_mod.mat');

for i=1:100
    dlmwrite(['images_rgb/sowerby_' int2str(i)] , a.data(i).image, 'delimiter', ' ');
    dlmwrite(['labels_raw/sowerby_' int2str(i)] , a.data(i).label, 'delimiter', ' ');
end

