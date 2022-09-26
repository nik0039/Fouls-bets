import joblib
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def clasterization(settings, cup):
    df1 = pd.read_csv('FoulsDataset_Season_' + cup + '_2021-2022_SofaScore.csv')
    km_silhouette = []
    stats = ['Team1', 'Team2', 'Team1_Ball_possession', 'Team2_Ball_possession', 'Team1_Yellow_cards',
             'Team2_Yellow_cards', 'Team1_PPDA', 'Team2_PPDA',
             'Team1_Passes', 'Team2_Passes', 'Team1_Long_balls', 'Team2_Long_balls', 'Team1_Crosses', 'Team2_Crosses',
             'Team1_Dribbles', 'Team2_Dribbles', 'Team1_Total_shots', 'Team2_Total_shots',
             'Team1_Shots_on_target', 'Team2_Shots_on_target', 'Team1_Shots_inside_box', 'Team2_Shots_inside_box',
             'Team1_Shots_outside_box', 'Team2_Shots_outside_box',
             'Team1_Duels_won', 'Team2_Duels_won', 'Team1_Aerials_won', 'Team2_Aerials_won',
             'Team1_Tackles', 'Team2_Tackles', 'Team1_Fouls', 'Team2_Fouls',
             'Team1_Corner_kicks', 'Team2_Corner_kicks', 'Team1_Interceptions', 'Team2_Interceptions',
             'Team1_Possession_lost', 'Team2_Possession_lost']
    if settings['Intensive']:
        stats.append('Team1_Intensive')
        stats.append('Team2_Intensive')
    if len(settings['Delete']) > 0:
        for delete_field in settings['Delete']:
            try:
                stats.remove("Team1_" + delete_field)
                stats.remove("Team2_" + delete_field)
            except ValueError:
                print("Не правильно выбраны поля на удаление")
    stat_teams = df1[stats]
    cols = ['Team_Ball_possession', 'Team_Yellow_cards',
            'Team_Passes', 'Team_Long_balls', 'Team_Crosses',
            'Team_Dribbles', 'Team_Total_shots', 'Team_Shots_on_target',
            'Team_Shots_inside_box', 'Team_Shots_outside_box',
            'Team_Duels_won', 'Team_Aerials_won',
            'Team_Tackles', 'Team_Fouls',
            'Team_Corner_kicks', 'Team_Interceptions',
            'Team_Possession_lost', 'Team_PPDA']
    if len(settings['Delete']) > 0:
        for delete_field in settings['Delete']:
            try:
                cols.remove("Team_" + delete_field)
            except ValueError:
                print("Не правильно выбраны поля на удаление")

    teams_full = pd.DataFrame(columns=cols)
    if settings['Intensive']:
        teams_full['Team_Intensive'] = ''

    teams = df1['Team1'].unique()
    teams = np.sort(teams)
    for team in teams:
        team_stata_home = stat_teams.loc[
            (stat_teams['Team1'] == team), list(filter(lambda x: x.find('Team1') != -1, stats))]
        # только содержащие Team1 / игры первой дома
        filtered = list(filter(lambda x: x.find('Team1') != -1, stats))
        # перед объединением приводим названия к единому стилю
        t1 = team_stata_home.rename(columns=dict(zip(filtered, map(lambda x: x.replace('1', ''), filtered))))
        team_stata_away = stat_teams.loc[
            (stat_teams['Team2'] == team), list(filter(lambda x: x.find('Team2') != -1, stats))]
        # только содержащие Team2 / игры первой на выезде
        filtered = list(filter(lambda x: x.find('Team2') != -1, stats))
        # перед объединением приводим названия к единому стилю
        t2 = team_stata_away.rename(columns=dict(zip(filtered, map(lambda x: x.replace('2', ''), filtered))))
        team_stata = pd.concat([t1, t2])
        team_stata = team_stata.sort_index()
        teams_full = teams_full.append(team_stata.mean(numeric_only=True), ignore_index=True)

    x_scale = teams_full
    for i in range(2, len(x_scale)):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(x_scale)
        preds = km.predict(x_scale)

        silhouette = silhouette_score(x_scale, preds)
        km_silhouette.append(silhouette)

    print(km_silhouette.index(max(km_silhouette)) + 2)

    clusters = settings['n_clusters']
    model = KMeans(n_clusters=clusters, random_state=1)
    model.fit(x_scale)
    teams_full['Type'] = model.predict(x_scale)
    teams_full['Team'] = teams
    return teams_full


def main():
    print("Чемпионат: ")
    champs = {1: 'РПЛ', 2: 'АПЛ', 3: 'Бундеслига', 4: 'Ла Лига', 5: 'Серия А', 6: 'Лига 1'}
    print(champs)
    Cup = champs[int(input())]

    df = pd.read_csv(f'FoulsDataset_Season_{Cup}_2021-2022_SofaScore.csv')
    df.Coef_1 = df.Coef_1.round(2)
    df.Coef_2 = df.Coef_2.round(2)
    teams = df['Team1'].unique()
    teams = np.sort(teams)
    teams_dict = {}
    for count, value in enumerate(teams, start=1):
        teams_dict[count] = value
    referees = df['Referee'].unique()
    referees = np.sort(referees)
    referees_dict = {}
    for count, value in enumerate(referees, start=1):
        referees_dict[count] = value
    print(teams_dict)

    print("Команда 1: ")
    Team1_id = int(input())
    print("Команда 2 : ")
    Team2_id = int(input())
    print(referees_dict)
    print("Судья: ")
    ref_id = int(input())
    print("П1: ")
    x1 = float(input())
    print("Х: ")
    x2 = float(input())
    print("П2: ")
    x3 = float(input())

    Team1 = teams_dict[Team1_id]
    Team2 = teams_dict[Team2_id]
    # Статистика судьи
    ref = referees_dict[ref_id]

    stats = ['Team1_Ball_possession', 'Team2_Ball_possession', 'Team1_Yellow_cards', 'Team2_Yellow_cards',
             'Team1_Passes', 'Team2_Passes', 'Team1_Long_balls', 'Team2_Long_balls', 'Team1_Crosses', 'Team2_Crosses',
             'Team1_Dribbles', 'Team2_Dribbles',
             'Team1_Intensive', 'Team2_Intensive', 'Team1_Total_shots', 'Team2_Total_shots',
             'Team1_Shots_inside_box', 'Team2_Shots_inside_box', 'Team1_PPDA', 'Team2_PPDA',
             'Team1_Duels_won', 'Team2_Duels_won', 'Team1_Aerials_won', 'Team2_Aerials_won',
             'Team1_Tackles', 'Team2_Tackles',
             'Team1_Corner_kicks', 'Team2_Corner_kicks', 'Team1_Interceptions', 'Team2_Interceptions',
             'Team1_Possession_lost', 'Team2_Possession_lost', 'Team1', 'Team2', 'Team1_Fouls', 'Team2_Fouls', 'Coef_1',
             'Coef_X', 'Coef_2']
    # кластеризация по стилю
    settings = {'РПЛ': {'n_clusters': 2,
                        'Intensive': True,
                        'Delete': []},
                'АПЛ': {'n_clusters': 3,
                        'Intensive': False,
                        'Delete': []},
                'Серия А': {'n_clusters': 2,
                            'Intensive': False,
                            'Delete': []},
                'Ла Лига': {'n_clusters': 3,
                            'Intensive': False,
                            'Delete': []},
                'Бундеслига': {'n_clusters': 3,
                               'Intensive': False,
                               'Delete': ['Long_balls', 'Possession_lost']},
                'Лига 1': {'n_clusters': 3,
                           'Intensive': False,
                           'Delete': []}}
    types_of_teams = clasterization(settings[Cup], Cup)
    print(types_of_teams[['Team', 'Type']].sort_values(by=['Type']))
    stat_referee = df[['Referee', 'Total_Fouls', 'Team1', 'Team2', 'Team1_Yellow_cards', 'Team2_Yellow_cards']]
    ref_stat = stat_referee.loc[(stat_referee['Referee'] == ref)]
    for mode in [1, 2]:
        print("Режим: ", mode)
        # Статистика команд
        stat_teams = df[stats]

        if len(ref_stat) == 0:
            ref_stat = stat_referee.loc[(stat_referee['Referee'] == ref)]
        # Все про домашнюю команду
        if mode == 1:
            team_stata_home = stat_teams.loc[
                (stat_teams['Team1'] == Team1), list(filter(lambda st: st.find('Team1') != -1, stats))]
        elif mode == 2:
            similiar_teams = list(
                types_of_teams.loc[
                    (types_of_teams['Type'] == int(types_of_teams.loc[(types_of_teams['Team'] == Team2)]['Type']))][
                    'Team'])
            team_stata_home = stat_teams.loc[
                (stat_teams['Team1'] == Team1) & (stat_teams['Team2'].isin(similiar_teams)), list(
                    filter(lambda st: st.find('Team1') != -1, stats))]

        # только содержащие Team1 / игры первой дома
        filtered = list(filter(lambda st: st.find('Team1') != -1, stats))
        # перед объединением приводим названия к единому стилю
        t1 = team_stata_home.rename(columns=dict(zip(filtered, map(lambda st: st.replace('1', ''), filtered))))

        if mode == 1:
            team_stata_away = stat_teams.loc[
                (stat_teams['Team2'] == Team1), list(filter(lambda st: st.find('Team2') != -1, stats))]
        elif mode == 2:
            similiar_teams1 = list(
                types_of_teams.loc[
                    (types_of_teams['Type'] == int(types_of_teams.loc[(types_of_teams['Team'] == Team2)]['Type']))][
                    'Team'])
            team_stata_away = stat_teams.loc[
                (stat_teams['Team2'] == Team1) & (stat_teams['Team1'].isin(types_of_teams)), list(
                    filter(lambda st: st.find('Team2') != -1, stats))]

        # только содержащие Team2 / игры первой на выезде
        filtered = list(filter(lambda st: st.find('Team2') != -1, stats))
        # перед объединением приводим названия к единому стилю
        t2 = team_stata_away.rename(columns=dict(zip(filtered, map(lambda st: st.replace('2', ''), filtered))))
        team_stata1 = pd.concat([t1, t2])
        # Средние общие
        team1_mean = team_stata1['Team_Fouls'].mean()
        team1_mean15 = team_stata1['Team_Fouls'].sort_index().tail(10).mean()  # 15

        team_stata1 = team_stata1.sort_index()
        # team_stata1 = team_stata1.sort_index().tail(15)

        # Средние домашние
        t1_mfh = t1['Team_Fouls'].mean()

        if pd.isna(t1_mfh):
            t1_mfh = team1_mean

        # Все про гостевую команду
        if mode == 1:
            team_stata_home = stat_teams.loc[
                (stat_teams['Team1'] == Team2), list(filter(lambda st: st.find('Team1') != -1, stats))]
        elif mode == 2:
            similiar_teams = list(
                types_of_teams.loc[
                    (types_of_teams['Type'] == int(types_of_teams.loc[(types_of_teams['Team'] == Team1)]['Type']))][
                    'Team'])
            team_stata_home = stat_teams.loc[
                (stat_teams['Team1'] == Team2) & (stat_teams['Team2'].isin(similiar_teams)), list(
                    filter(lambda st: st.find('Team1') != -1, stats))]
            # только содержащие Team1 / игры второй дома
        filtered = list(filter(lambda st: st.find('Team1') != -1, stats))
        # перед объединением приводим названия к единому стилю
        t1 = team_stata_home.rename(columns=dict(zip(filtered, map(lambda st: st.replace('1', ''), filtered))))

        if mode == 1:
            team_stata_away = stat_teams.loc[
                (stat_teams['Team2'] == Team2), list(filter(lambda st: st.find('Team2') != -1, stats))]
        elif mode == 2:
            similiar_teams2 = list(
                types_of_teams.loc[
                    (types_of_teams['Type'] == int(types_of_teams.loc[(types_of_teams['Team'] == Team1)]['Type']))][
                    'Team'])
            team_stata_away = stat_teams.loc[
                (stat_teams['Team2'] == Team2) & (stat_teams['Team1'].isin(similiar_teams2)), list(
                    filter(lambda st: st.find('Team2') != -1, stats))]

        # только содержащие Team2 / игры второй на выезде
        filtered = list(filter(lambda st: st.find('Team2') != -1, stats))
        # перед объединением приводим названия к единому стилю
        t2 = team_stata_away.rename(columns=dict(zip(filtered, map(lambda st: st.replace('2', ''), filtered))))
        team_stata2 = pd.concat([t1, t2])
        # Средние общие
        team2_mean = team_stata2['Team_Fouls'].mean()
        team2_mean15 = team_stata2['Team_Fouls'].sort_index().tail(10).mean()  # .tail(10)

        team_stata2 = team_stata2.sort_index()
        # team_stata2 = team_stata2.sort_index().tail(15)

        # Средние домашние
        t2_mfa = t2['Team_Fouls'].mean()

        if pd.isna(t2_mfa):
            t2_mfa = team2_mean

        # берем статистики без номера, так как в team_stata используются без номеров
        stats_num = stats[:len(stats) - 7]

        filtered = list(filter(lambda st: st.find('Team1') != -1, stats_num))
        stats_without_num = list(map(lambda st: st.replace('1', ''), filtered))
        mean_stats = []
        for i in stats_without_num:
            mean_stats.append([team_stata1[i].mean()])
            mean_stats.append([team_stata2[i].mean()])

        final_dict = {'Coef_1': [x1], 'Coef_X': [x2], 'Coef_2': [x3],
                      'Ref_Mean_Fouls': [ref_stat['Total_Fouls'].mean()],
                      'Team1_Mean_Fouls_Home': [t1_mfh], 'Team2_Mean_Fouls_Away': [t2_mfa],
                      'Ref_Mean_Cards': [ref_stat[['Team1_Yellow_cards', 'Team2_Yellow_cards']].sum(axis=1).mean()], }

        stat_dict = dict(zip(stats_num, mean_stats))
        final_dict.update(stat_dict)

        x = pd.DataFrame(final_dict)
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(x)
        statistics = ['Intensive', 'Yellow_cards', 'Passes', 'Long_balls', 'Crosses', 'Dribbles', 'PPDA',
                      'Duels_won', 'Aerials_won', 'Tackles', 'Corner_kicks', 'Interceptions', 'Shots_inside_box',
                      'Total_shots', 'Possession_lost']
        for stat in statistics:
            cols = ["Team1_" + stat, "Team2_" + stat]
            x = x.eval("{stat} = {}".format("+".join(cols), stat=stat)).drop(columns=cols)
        help_dict = {'1': Team1, '2': Team2}
        for_delete = ['2', '1']
        pd.set_option('display.max_columns', None)

        for key in help_dict:
            if mode == 1:
                t = df.loc[
                    (df['Team' + key] == help_dict[key]), ['Team1', 'Team1_PPDA', 'Coef_1', 'Team1_Intensive',
                                                           'Team1_Fouls', 'Team2',
                                                           'Team2_PPDA', 'Coef_2', 'Team2_Intensive', 'Team2_Fouls',
                                                           'Total_Fouls']].sort_values(by='Total_Fouls')
                t['Sum_PPDA'] = t.Team1_PPDA + t.Team2_PPDA
                print(t)
                print('------------------------------------------------------------------------')
            else:
                if key == '1':
                    similiar_teams = similiar_teams1
                else:
                    similiar_teams = similiar_teams2
                t = df.loc[
                    (df['Team' + key] == help_dict[key]) & (df['Team' + for_delete[0]].isin(similiar_teams)),
                    ['Team1', 'Team1_PPDA', 'Coef_1', 'Team1_Intensive', 'Team1_Fouls',
                     'Team2', 'Team2_PPDA', 'Coef_2', 'Team2_Intensive', 'Team2_Fouls', 'Total_Fouls']].sort_values(
                    by='Total_Fouls')
                for_delete.pop(0)
                t['Sum_PPDA'] = t.Team1_PPDA + t.Team2_PPDA
                print(t)
                print('------------------------------------------------------------------------')

        regressor = joblib.load(f"\\Модели\\Linear Regression {Cup}.pkl")

        elastic_net = joblib.load(f"\\Модели\\ElasticNet {Cup}.pkl")
        bayesian_ridge = joblib.load(f"\\Модели\\Bayesian Ridge {Cup}.pkl")
        huber_reg = joblib.load(f"\\Модели\\Huber Regression {Cup}.pkl")
        kr = joblib.load(f"\\Модели\\Kernel Ridge {Cup}.pkl")
        theil_sen = joblib.load(f"\\Модели\\TheilSen {Cup}.pkl")

        models = [regressor, elastic_net, bayesian_ridge, huber_reg, kr, theil_sen]
        table = PrettyTable()
        table.field_names = ['Линейная регрессия', 'Эластичная сеть', 'Байес', 'Huber', 'Kernel Ridge', 'TheilSen']
        row = []
        for model in models:
            row.append(str(model.predict(x)).replace("[", "").replace("]", "").replace(".", ","))
        table.add_row(row)
        table.vertical_char = ' '

        print('Средние общие: ', team1_mean + team2_mean)
        print('Средние общие 10 последних: ', team1_mean15 + team2_mean15)
        print('Средние дом//выезд: ', t1_mfh + t2_mfa)
        print(table)
        print('Линейная регрессия: ', regressor.predict(x))
        print('Эластичная сеть: ', elastic_net.predict(x))
        print('Байес: ', bayesian_ridge.predict(x))
        print('Huber: ', huber_reg.predict(x))
        print('Kernel Ridge: ', kr.predict(x))
        print('TheilSen: ', theil_sen.predict(x))

        print('------------------------------------Общий------------------------------------')

        # -------------------------------------------------------------------------------------------------------
        from sklearn.linear_model import LogisticRegression

        print('-----------------------------------------------------------------------')
        print('Фол для анализа:')

        df1 = pd.read_csv(r'Датасеты\FoulsDataset_Season_' + Cup + '_2019-2020_SofaScore.csv')
        df2 = pd.read_csv(r'Датасеты\FoulsDataset_Season_' + Cup + '_2020-2021_SofaScore.csv')
        df3 = pd.read_csv(r'Датасеты\FoulsDataset_Season_' + Cup + '_2021-2022_SofaScore.csv')
        df_train = pd.concat([df1, df2, df3], ignore_index=True)

        books_fouls = float(input())
        while books_fouls != 0:
            for i, row in df_train.iterrows():
                if df_train['Total_Fouls'][i] > books_fouls:
                    df_train.loc[i, 'LessOrMore'] = 1
                else:
                    df_train.loc[i, 'LessOrMore'] = 0

            x_lr = df_train[
                ['Coef_1', 'Coef_X', 'Coef_2', 'Ref_Mean_Fouls', 'Team1_Mean_Fouls_Home', 'Team2_Mean_Fouls_Away',
                 'Ref_Mean_Cards', 'Team1_Intensive', 'Team2_Intensive', 'Team1_Total_shots', 'Team2_Total_shots',
                 'Team1_Shots_inside_box', 'Team2_Shots_inside_box', 'Team1_PPDA', 'Team2_PPDA',
                 'Team1_Ball_possession', 'Team2_Ball_possession',
                 'Team1_Yellow_cards', 'Team2_Yellow_cards', 'Team1_Passes', 'Team2_Passes',
                 'Team1_Long_balls', 'Team2_Long_balls', 'Team1_Crosses', 'Team2_Crosses',
                 'Team1_Dribbles', 'Team2_Dribbles', 'Team1_Duels_won', 'Team2_Duels_won',
                 'Team1_Aerials_won', 'Team2_Aerials_won', 'Team1_Tackles', 'Team2_Tackles',
                 'Team1_Corner_kicks', 'Team2_Corner_kicks', 'Team1_Interceptions', 'Team2_Interceptions',
                 'Team1_Possession_lost', 'Team2_Possession_lost']]
            statistics = ['Intensive', 'Yellow_cards', 'Passes', 'Long_balls', 'Crosses', 'Dribbles', 'PPDA',
                          'Duels_won', 'Aerials_won', 'Tackles', 'Corner_kicks', 'Interceptions', 'Shots_inside_box',
                          'Total_shots', 'Possession_lost']
            for stat in statistics:
                cols = ["Team1_" + stat, "Team2_" + stat]
                x_lr = x_lr.eval("{stat} = {}".format("+".join(cols), stat=stat)).drop(columns=cols)
            y = df_train['LessOrMore']
            y = y.astype('int')

            lr = LogisticRegression(random_state=42, max_iter=1000000)
            lr.fit(x_lr, y)

            print(lr.predict(x))
            print(lr.predict_proba(x))

            print('Фол для анализа:')
            books_fouls = float(input())


if __name__ == "__main__":
    main()
