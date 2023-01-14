clear all

% load data
load('OCT_accuracy_non_iid.mat')
% Baseline 1: CL
data_1 = OCT_accuracy_non_iid(:,1);
% Baseline 2: FL
data_2 = OCT_accuracy_non_iid(:,2);
% Sequential FedSL
data_3 = OCT_accuracy_non_iid(:,3);
% Parallel FedSL
data_4 = OCT_accuracy_non_iid(:,4);

% plot figure
figure

% draw curves with shadows
curve_1 = plot(data_1,'b-','Linewidth',1.5)
hold on
curve_2 = plot(data_2,'r-','Linewidth',1.5)
curve_3 = plot(data_3,'m-','Linewidth',1.5)
curve_4 = plot(data_4,'c-','Linewidth',1.5)

curve_1.Color(4) = 0.2;
curve_2.Color(4) = 0.2;
curve_3.Color(4) = 0.2;
curve_4.Color(4) = 0.2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% draw mean curves
data_5 = average_data(data_1);
data_6 = average_data(data_2);
data_7 = average_data(data_3);
data_8 = average_data(data_4);

curve_5 = plot(data_5,'b-o','Linewidth',1.5,'MarkerIndices',[50 100 150 200 250 300 350])
hold on
curve_6 = plot(data_6,'r-v','Linewidth',1.5,'MarkerIndices',[50 100 150 200 250 300 350])
curve_7 = plot(data_7,'m-*','Linewidth',1.5,'MarkerIndices',[50 100 150 200 250 300 350])
curve_8 = plot(data_8,'c-s','Linewidth',1.5,'MarkerIndices',[50 100 150 200 250 300 350])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

title('OCT Images: Non-IID','FontWeight','normal')
xlabel('Training rounds')
ylabel('Prediction accuracy (%)')
set(gca,'FontName','Times New Roman','FontSize',14);
legend_1 = legend([curve_5 curve_6 curve_7 curve_8], ...
                  'Baseline 1: CL',...
                  'Baseline 2: FL',...
                  'Sequential FedSL',...
                  'Parallel FedSL',...
                  'Location','west');
set(legend_1,'FontSize',14);
xlim([0 400])
ylim([20 90])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% draw subfigure within a given position
axes('position',[0.51,0.2,0.37,0.45]);

% draw curves with shadows
curve_1 = plot(data_1,'b-','Linewidth',1.5)
hold on
curve_2 = plot(data_2,'r-','Linewidth',1.5)
curve_3 = plot(data_3,'m-','Linewidth',1.5)
curve_4 = plot(data_4,'c-','Linewidth',1.5)

curve_1.Color(4) = 0.2;
curve_2.Color(4) = 0.2;
curve_3.Color(4) = 0.2;
curve_4.Color(4) = 0.2;

% draw mean curves
curve_5 = plot(data_5,'b-o','Linewidth',1.5,'MarkerIndices',[1 10 20 40 60 80 100])
curve_6 = plot(data_6,'r-v','Linewidth',1.5,'MarkerIndices',[1 10 20 40 60 80 100])
curve_7 = plot(data_7,'m-*','Linewidth',1.5,'MarkerIndices',[1 10 20 40 60 80 100])
curve_8 = plot(data_8,'c-s','Linewidth',1.5,'MarkerIndices',[1 10 20 40 60 80 100])

title('')
xlabel('')
ylabel('')
xlim([0 100])
ylim([20 90])

function data_avg = average_data(x)
    K = 400;
    data_avg = zeros(K,1);
    for i = 1:K
        data_avg(i) = mean(x(i:i+3));
    end
end

