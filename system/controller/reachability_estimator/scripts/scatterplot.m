function []=scatterplot(input_mat, output_pgf, output_png)

T = load(input_mat);
labels = cellstr(T.x);
labels = labels(1:5:end);
labels_num = repelem(1:11, 5);

scatter(T.y, labels_num, "filled")

xlim([0 1])
xlabel('F1-score')

yticks(1:11)
yticklabels(labels)
ylabel('Configurations')
saveas (1, output_pgf, 'tikz')
saveas (1, output_png, 'png')

end
