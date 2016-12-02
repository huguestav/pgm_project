a = load('corel_subset.mat');
cmap = colormap('gray(5)');
for i=1:100
    imwrite(a.data(i).image, ['Images/corel_' int2str(i) '.jpg'])
    imwrite(a.data(i).label,cmap, ['Labels/corel_' int2str(i) '.jpg'])
end
