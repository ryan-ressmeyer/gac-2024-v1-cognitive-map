%%% Plot MUA - GAC dataset
% 2025 P. Papale fecit

clear all

% constants
datadir_gen = '/home/ryanress/code/Data_for_paper/';
monkeys = {'monkeyF';'monkeyN'};
% monkeys = {'monkeyN'};
% tasks = {'lums','sacc','length','thick'};
tasks = {'lums','sacc'};
% tasks = {'sacc'};
snr_th = 1;

for m = 1:length(monkeys)
    monkey = monkeys{m};
    if monkey == 'monkeyF'
        array_chns = 129:192;
        all_chns = zeros([1 512])';
        all_chns(array_chns) = 1;
    else
        array_chns = 193:256;
        all_chns = zeros([1 512])';
        all_chns(array_chns) = 1;
    end
    % task = 'lines';
    for t = 1:length(tasks)
        clear task filename ALLMAT normMUA SNR lats
        task = tasks{t};
        filename = [datadir_gen,monkey,'/ObjAtt_GAC2_',task,'_MUA_trials.mat'];
        load(filename,'ALLMAT','tb')
        tb(end+1) = tb(end)+1;
        clear filename
        filename = [datadir_gen,monkey,'/ObjAtt_GAC2_',task,'_normMUA.mat'];
        load(filename)
        control_var = 3;
        switch task
            case 'lums'
                control_var = 8;
                control_combs = [1,3;2,5;3,1;4,6;5,2;6,4];
            case 'sacc'
                control_var = 9;
                control_combs = [1,1;2,2;3,3;4,4];
        end
        controls = unique(ALLMAT(:,control_var));
        sum(SNR>snr_th & all_chns==1)
        
        figure
        plot(smooth(squeeze(nanmean(normMUA(SNR>snr_th & all_chns==1,ALLMAT(:,4)==2,:),[1,2])),20),'Color',[.7 .6 .8],'LineWidth',2)
        hold on
        plot(smooth(squeeze(nanmean(normMUA(SNR>snr_th & all_chns==1,ALLMAT(:,4)==1,:),[1,2])),20),'Color',[.3 .7 .2],'LineWidth',2)
        xlim([0 701])
        ylim([-.3 1.1])
        line([200 200],[-.3 1.1],'color',[0.5 .5 .5],'LineStyle','--','LineWidth',1)
        set(gca,'XTick',1:100:size(tb,2))
        set(gca,'XTickLabel',round(tb((1:100:end)),2))
        set(gca,'YTick',[0,1])
        ylabel('Normalized activity')
        xlabel({'Time','(ms)'})
        box off
        daspect([500 .5 1])

%         figure
%         deh = squeeze(nanmean(normMUA(SNR>snr_th & all_chns==1,ALLMAT(:,4)==1,:),2));
%         deh1 = squeeze(nanmean(normMUA(SNR>snr_th & all_chns==1,ALLMAT(:,4)==2,:),2));
%         for i = 1:size(deh,1)
%             subplot(7,7,i)
%             plot(deh(i,:));
%             hold on
%             plot(deh1(i,:));
%         end
        figure
        z=1;
        for c = 1:length(controls)
            subplot(1,length(controls),z)
            plot(smooth(squeeze(nanmean(normMUA(SNR>snr_th & all_chns==1,ALLMAT(:,4)==2 & ALLMAT(:,control_var)==control_combs(c,2),:),[1,2])),20),'Color',[.7 .6 .8],'LineWidth',2)
            hold on
            plot(smooth(squeeze(nanmean(normMUA(SNR>snr_th & all_chns==1,ALLMAT(:,4)==1 & ALLMAT(:,control_var)==control_combs(c,1),:),[1,2])),20),'Color',[.3 .7 .2],'LineWidth',2)
            z = z+1;
            ylim([-.3 1.4])
            xlim([0 701])
            line([200 200],[-.3 1.4],'color',[0.5 .5 .5],'LineStyle','--','LineWidth',1)
            set(gca,'XTick',1:100:size(tb,2))
            set(gca,'XTickLabel',round(tb((1:100:end)),2))
            set(gca,'YTick',[0,1])
            ylabel('Normalized activity')
            xlabel({'Time','(ms)'})
            box off
            daspect([500 .5 1])
        end
        drawnow
    end
end