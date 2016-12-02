a = load('sowerby_mod.mat');
cmap = colormap('hsv(7)');
for i=1:104
    imwrite(a.data(i).image, ['Images/sowerby_' int2str(i) '.jpg'])
    imwrite(a.data(i).label,cmap, ['Labels/sowerby_' int2str(i) '.jpg'])
end
