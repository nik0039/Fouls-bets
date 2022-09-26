import pandas as pd
import time
from random import randint
from deep_translator import GoogleTranslator
import cloudscraper
import re

if __name__ == "__main__":
    urls = {'РПЛ': {'2019-2020': 'https://api.sofascore.com/api/v1/unique-tournament/203/season/23682/events/round/',
                    '2020-2021': 'https://api.sofascore.com/api/v1/unique-tournament/203/season/29200/events/round/',
                    '2021-2022': 'https://api.sofascore.com/api/v1/unique-tournament/203/season/37038/events/round/',
                    '2022-2023': 'https://api.sofascore.com/api/v1/unique-tournament/203/season/42388/events/round/',
                    'tours': [30, 8]},
            'АПЛ': {'2019-2020': 'https://api.sofascore.com/api/v1/unique-tournament/17/season/23776/events/round/',
                    '2020-2021': 'https://api.sofascore.com/api/v1/unique-tournament/17/season/29415/events/round/',
                    '2021-2022': 'https://api.sofascore.com/api/v1/unique-tournament/17/season/37036/events/round/',
                    '2022-2023': 'https://api.sofascore.com/api/v1/unique-tournament/17/season/41886/events/round/',
                    'tours': [38, 10]},
            'Серия А': {'2019-2020': 'https://api.sofascore.com/api/v1/unique-tournament/23/season/24644/events/round/',
                        '2020-2021': 'https://api.sofascore.com/api/v1/unique-tournament/23/season/32523/events/round/',
                        '2021-2022': 'https://api.sofascore.com/api/v1/unique-tournament/23/season/37475/events/round/',
                        '2022-2023': 'https://api.sofascore.com/api/v1/unique-tournament/23/season/42415/events/round/',
                        'tours': [38, 10]},
            'Ла Лига': {'2019-2020': 'https://api.sofascore.com/api/v1/unique-tournament/8/season/24127/events/round/',
                        '2020-2021': 'https://api.sofascore.com/api/v1/unique-tournament/8/season/32501/events/round/',
                        '2021-2022': 'https://api.sofascore.com/api/v1/unique-tournament/8/season/37223/events/round/',
                        '2022-2023': 'https://api.sofascore.com/api/v1/unique-tournament/8/season/42409/events/round/',
                        'tours': [38, 10]},
            'Бундеслига': {
                '2019-2020': 'https://api.sofascore.com/api/v1/unique-tournament/35/season/23538/events/round/',
                '2020-2021': 'https://api.sofascore.com/api/v1/unique-tournament/35/season/28210/events/round/',
                '2021-2022': 'https://api.sofascore.com/api/v1/unique-tournament/35/season/37166/events/round/',
                '2022-2023': 'https://api.sofascore.com/api/v1/unique-tournament/35/season/42268/events/round/',
                'tours': [34, 9]},
            'Лига 1': {'2019-2020': 'https://api.sofascore.com/api/v1/unique-tournament/34/season/23872/events/round/',
                       '2020-2021': 'https://api.sofascore.com/api/v1/unique-tournament/34/season/28222/events/round/',
                       '2021-2022': 'https://api.sofascore.com/api/v1/unique-tournament/34/season/37167/events/round/',
                       '2022-2023': 'https://api.sofascore.com/api/v1/unique-tournament/34/season/42273/events/round/',
                       'tours': [38, 10]}
            }
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/50.0.2661.102 '
            'Safari/537.36'}
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'firefox',
            'platform': 'windows',
            'mobile': False
        }
    )
    years = ['2022-2023']

    statistics = ['Ball possession', 'Fouls', 'Yellow cards', 'Passes', 'Long balls', 'Crosses', 'Dribbles',
                  'Duels won', 'Aerials won', 'Tackles', 'Corner kicks', 'Interceptions', 'Possession lost',
                  'Total shots', 'Shots on target', 'Shots inside box', 'Shots outside box']
    statisticsItem = []
    statistics1 = ['Team1', 'Team2', 'Referee', 'Coef_1', 'Coef_X', 'Coef_2', 'Total_Fouls', 'Team1_Score',
                   'Team2_Score', 'Team1_Intensive', 'Team2_Intensive', 'Intensive']
    for k in statistics:
        k = k.replace(' ', '_')
        statistics1.append('Team1_' + k)
        statistics1.append('Team2_' + k)

    df = pd.DataFrame(columns=statistics1)
    finalDictionary = dict()

    keys = ['РПЛ', 'АПЛ', 'Серия А', 'Бундеслига', 'Лига 1', 'Ла Лига']
    for key in keys:
        for year in years:
            df = df.iloc[0:0]
            for i in range(1, urls[key]["tours"][0] + 1):
                c = 0
                print('Тур ' + str(i))

                Tour = scraper.get(urls[key][year] + str(i)).json()
                time.sleep(randint(1, 2))
                for count in range(len(Tour['events'])):
                    for w in statistics1:
                        finalDictionary[w] = None
                    if Tour['events'][count]['status']['description'] == 'Ended' and 'isAwarded' not in Tour['events'][
                        count]:
                        finalDictionary['Team1'] = GoogleTranslator(source='english', target='russian').translate(
                            'FC ' + Tour['events'][count]['homeTeam']['shortName']).replace('ФК ', '')
                        finalDictionary['Team2'] = GoogleTranslator(source='english', target='russian').translate(
                            'FC ' + Tour['events'][count]['awayTeam']['shortName']).replace('ФК ', '')

                        # добавить рефери ↓
                        Total = scraper.get(
                            'https://api.sofascore.com/api/v1/event/' + str(Tour['events'][count]['id'])).json()
                        finalDictionary['Referee'] = GoogleTranslator(source='english', target='russian').translate(
                            Total['event']['referee']['name'])
                        finalDictionary['Team1_Score'] = int(Total['event']['homeScore']['normaltime'])
                        finalDictionary['Team2_Score'] = int(Total['event']['awayScore']['normaltime'])
                        Graph = scraper.get(
                            'https://api.sofascore.com/api/v1/event/' + str(
                                Tour['events'][count]['id']) + '/graph').json()
                        Team1_Intensive = 0
                        Team2_Intensive = 0
                        if 'graphPoints' in Graph:
                            for minute in Graph['graphPoints']:
                                if minute['value'] > 0:
                                    Team1_Intensive += minute['value']
                                else:
                                    Team2_Intensive += minute['value']
                        else:
                            Team1_Intensive = 0
                            Team2_Intensive = 0
                        finalDictionary['Team1_Intensive'] = abs(Team1_Intensive)
                        finalDictionary['Team2_Intensive'] = abs(Team2_Intensive)
                        finalDictionary['Intensive'] = Team1_Intensive + Team2_Intensive
                        odds = scraper.get(
                            'https://api.sofascore.com/api/v1/event/' + str(Tour['events'][count]['id']) + '/odds/1/all'
                        ).json()
                        time.sleep(randint(1, 2))
                        if 'markets' in odds:
                            arrOdds = odds['markets'][0]['choices'][0]['fractionalValue'].replace('\\', '').split('/')
                            finalDictionary['Coef_1'] = ((int(arrOdds[0]) / int(arrOdds[1])) + 1)
                            arrOdds = odds['markets'][0]['choices'][1]['fractionalValue'].replace('\\', '').split('/')
                            finalDictionary['Coef_X'] = ((int(arrOdds[0]) / int(arrOdds[1])) + 1)
                            arrOdds = odds['markets'][0]['choices'][2]['fractionalValue'].replace('\\', '').split('/')
                            finalDictionary['Coef_2'] = ((int(arrOdds[0]) / int(arrOdds[1])) + 1)
                        else:
                            finalDictionary['Coef_1'] = 0
                            finalDictionary['Coef_X'] = 0
                            finalDictionary['Coef_2'] = 0
                        match = scraper.get(
                            'https://api.sofascore.com/api/v1/event/' + str(
                                Tour['events'][count]['id']) + '/statistics').json()
                        time.sleep(randint(1, 2))
                        statisticsItem.clear()
                        # match['statistics'][0]['groups'] статистики за весь матч
                        for stat in match['statistics'][0]['groups']:
                            for j in range(len(stat['statisticsItems'])):
                                statisticsItem.append(stat['statisticsItems'][j])
                        for stat in statistics:
                            select_stat = next((item for i, item in enumerate(statisticsItem) if item["name"] == stat),
                                               {'home': str(0), 'away': str(0)})
                            stat = stat.replace(' ', '_')
                            if select_stat['home'].find("/") == -1:
                                finalDictionary['Team1_' + stat] = int(select_stat['home'].replace('%', ''))
                                finalDictionary['Team2_' + stat] = int(select_stat['away'].replace('%', ''))
                            else:
                                finalDictionary['Team1_' + stat] = int(re.split("[/ ]", select_stat['home'])[1])
                                finalDictionary['Team2_' + stat] = int(re.split("[/ ]", select_stat["away"])[1])
                        for s in match['statistics'][0]['groups'][2]['statisticsItems']:
                            if s['name'] == "Fouls":
                                finalDictionary['Total_Fouls'] = int(s['home']) + int(s['away'])
                                break
                        pregame = scraper.get(
                            'https://api.sofascore.com/api/v1/event/' + str(
                                Tour['events'][count]['id']) + '/pregame-form').json()
                        try:
                            finalDictionary['Team1_Ratings'] = float(pregame['homeTeam']['avgRating'])
                            finalDictionary['Team2_Ratings'] = float(pregame['awayTeam']['avgRating'])
                        except:
                            finalDictionary['Team1_Ratings'] = 0
                            finalDictionary['Team2_Ratings'] = 0
                        lineups = scraper.get(
                            'https://api.sofascore.com/api/v1/event/' + str(
                                Tour['events'][count]['id']) + '/lineups').json()
                        finalDictionary['Team1_Formation'] = lineups['home']['formation']
                        finalDictionary['Team2_Formation'] = lineups['away']['formation']
                        df = df.append(finalDictionary, ignore_index=True)
                        time.sleep(randint(1, 2))
                        c += 1
                        print(c)
            df.to_csv(f'FoulsDataset_Season_{key}_{year}_SofaScore.csv')
