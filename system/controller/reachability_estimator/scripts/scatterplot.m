function []=scatterplot(input_mat, output_pgf)

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

disp("Hello World")

end
