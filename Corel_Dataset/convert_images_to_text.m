a = load('corel_subset.mat');

for i=1:100
    dlmwrite(['images_rgb/corel_' int2str(i)] , a.data(i).image, 'delimiter', ' ');
    dlmwrite(['labels_raw/corel_' int2str(i)] , a.data(i).label, 'delimiter', ' ');
end

