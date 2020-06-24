import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r"C:\Users\Áron\Desktop\Courses\csgo-ai-competition-master\dataset_initial")

X = pd.read_json("dataset_00.json")
df_X = X
y = pd.DataFrame(X.pop('round_winner').map(lambda s: 0 if s == "CT" else 1))


"""
                            1. DATA PREPROCESSING
"""

#Converting positions into 9 sectors
map_coords = {
    "de_cache":    (-2000, 3250),
    "de_dust2":    (-2476, 3239),
    "de_inferno":  (-2087, 3870),
    "de_mirage":   (-3230, 1713),
    "de_nuke":     (-3453, 2887),
    "de_overpass": (-4831, 1781),
    "de_train":    (-2477, 2392),
    "de_vertigo":  (-3168, 1762),
}

map_len = {
    "de_cache":    5500,
    "de_dust2":    4400,
    "de_inferno":  4900,
    "de_mirage":   5000,
    "de_nuke":     7000,
    "de_overpass": 5200,
    "de_train":    4700,
    "de_vertigo":  4000,
}

last_positions_ct = []
last_positions_t = []
for row in X["alive_players"]:
    for i in row:
        pos = i.get("position_history")
        if pos != []:
            if i.get("team") =="CT":
                last_positions_ct.append(pos[len(pos)-1])
            else:
                last_positions_t.append(pos[len(pos)-1])
    last_positions_t.append("next_row")
    last_positions_ct.append("next_row")
    

def get_sector(map_name, x, y):
    diff = map_len.get(map_name) / 3
    lower_limit_x = map_coords.get(map_name)[0]
    upper_limit_y = map_coords.get(map_name)[1]
    upper_limit_x = lower_limit_x + diff
    lower_limit_y = upper_limit_y - diff
    for row in range(1,4):
        lower_limit_x = map_coords.get(map_name)[0]
        upper_limit_x = lower_limit_x + diff
        lower_limit_y = upper_limit_y - diff
        for column in range(1,4):
            if x > lower_limit_x and x < upper_limit_x and y < upper_limit_y and y > lower_limit_y:
                return str(row)+str(column)
            lower_limit_x = upper_limit_x
            upper_limit_x += diff           
        upper_limit_y -= diff        
 
def setup_sectors(map_name, last_position):   
    if last_position == "next_row":
        return "next_row"
    else:
        return get_sector(map_name, last_position.get("x"), last_position.get("y"))
        

def sectors(last_positions_list):
    index = 0    
    sectors = []
    for i in last_positions_list:
        if setup_sectors(X["map"][index], i) == "next_row":
            sectors.append("next_row")
            index += 1
        else:
            sectors.append(setup_sectors(X["map"][index], i))
    return sectors

   
def sector_columns(sector_list):
    s1, s2, s3, s4, s5, s6, s7, s8, s9 = ([] for i in range(9))
    s1_count, s2_count, s3_count, s4_count, s5_count, s6_count, s7_count, s8_count, s9_count = (0 for i in range(9))
    for i in sector_list:
        if i == "next_row":
            s1.append(s1_count)
            s2.append(s2_count)
            s3.append(s3_count)
            s4.append(s4_count)
            s5.append(s5_count)
            s6.append(s6_count)
            s7.append(s7_count)
            s8.append(s8_count)
            s9.append(s9_count)
            s1_count, s2_count, s3_count, s4_count, s5_count, s6_count, s7_count, s8_count, s9_count = (0 for i in range(9))
        else:
            if i == "11":
                s1_count += 1
            elif i == "12":
                s2_count += 1
            elif i == "13":
                s3_count += 1
            elif i == "21":
                s4_count += 1
            elif i == "22":
                s5_count += 1
            elif i == "23":
                s6_count += 1
            elif i == "31":
                s7_count += 1
            elif i == "32":
                s8_count += 1
            elif i == "33":
                s9_count += 1
    return s1,s2,s3,s4,s5,s6,s7,s8,s9
 
for index in range(1, 10):
    X["CTs_in_sector_"+str(index)] = sector_columns(sectors(last_positions_ct))[index-1]
    X["Ts_in_sector_"+str(index)] = sector_columns(sectors(last_positions_t))[index-1]
       

#Transforming the score into leading team and lead num
def lead_round_diff():
    team = []
    diff = [] 
    for score in X["current_score"]:
        if score[0] > score[1]:
            team.append("CT")
            diff.append(score[0] - score[1])
        elif score[0] < score[1]:
            team.append("Terrorist")
            diff.append(score[1] - score[0])
        else:
            team.append("draw")
            diff.append(0)
    return team, diff
X["leading_team"] = lead_round_diff()[0]
X["round_difference"] = lead_round_diff()[1]
X.drop(["current_score"], axis = 1, inplace = True)

#Separating planted into none, A, B
bomb_plant = []
for row in X["planted_bomb"]:
    if row == None:
        bomb_plant.append("not_planted")
    elif row["site"] == "A":
        bomb_plant.append("planted_on_A")
    else:
        bomb_plant.append("planted_on_B")


#One hot encoding: map, round_status, leading_team
OHE_map = pd.get_dummies(X["map"], dtype= int) 
OHE_round_status = pd.get_dummies(X["round_status"], dtype= int, prefix = "round_status") 
OHE_leading_team = pd.get_dummies(X["leading_team"], dtype= int, prefix = "leading_team") 
OHE_bomb_plant = pd.get_dummies(bomb_plant, dtype= int)

X = pd.concat([X, OHE_map], axis = 1)
X = pd.concat([X, OHE_round_status], axis = 1)
X = pd.concat([X, OHE_leading_team], axis = 1)
X = pd.concat([X, OHE_bomb_plant], axis = 1)

X.drop(["map"],axis=1, inplace=True)
X.drop(["round_status"],axis=1, inplace=True)
X.drop(["leading_team"],axis=1, inplace=True)
X.drop(["planted_bomb"],axis=1, inplace=True)


#Firepower + utility info
def setup_inventory(team):
    inventory_list = []
    for roundn in X["alive_players"]:
        for players in roundn:
            for inventory in players["inventory"]:
                if players["team"] == team:
                    inventory_list.append(inventory)
        inventory_list.append("NEXT_ROW")
    return inventory_list


def detect_weapon(inventory_list, weapon):
    weapon_count = 0
    weapon_in_round = []
    for item in inventory_list:
        if item != "NEXT_ROW" and item["item_type"] == weapon:
            weapon_count += 1  
        elif item == "NEXT_ROW":
            weapon_in_round.append(weapon_count)
            weapon_count = 0
    return weapon_in_round
        

grenades = ["Flashbang", "HeGrenade", "SmokeGrenade", "IncendiaryGrenade", 
            "MolotovGrenade","DecoyGrenade"]
def detect_grenades(inventory_list):
    grenades_on_given_side = []
    grenade_count = 0
    for item in inventory_list:
        if item != "NEXT_ROW" and item["item_type"] in grenades:
            grenade_count += item["clip_ammo"]
        if item == "NEXT_ROW":
            grenades_on_given_side.append(grenade_count)
            grenade_count = 0
    return grenades_on_given_side


def active_utility_count(utility_column):
    active_util = []
    for item in X[utility_column]:
        active_util.append(len(item))
    return active_util


t_inventory = setup_inventory("Terrorist")
ct_inventory = setup_inventory("CT")          
X["t_awps"] = np.array(detect_weapon(t_inventory, "Awp"))
X["ct_awps"] = np.array(detect_weapon(ct_inventory, "Awp"))       
X["ct_grenades"] = np.array(detect_grenades(ct_inventory))
X["t_grenades"] = np.array(detect_grenades(t_inventory))
X["active_smokes_count"] = np.array(active_utility_count("active_smokes"))
X["active_molotovs_count"] = np.array(active_utility_count("active_molotovs"))
X.drop(["active_smokes"], axis = 1, inplace = True)
X.drop(["active_molotovs"], axis = 1, inplace = True) 


#Getting basics: money, health ect.
def get_attr(ds, team, attr=None):
    team_players = map(lambda players: filter(lambda p: p["team"] == team, players), 
                       ds['alive_players'])
    if attr:
        team_players = map(lambda players: map(lambda p: p[attr], players), team_players)
    
    return list(map(lambda p: list(p), team_players))

for ds in [X]:
    ds['alive_players_t']  = list(map(len ,get_attr(ds, "Terrorist")))
    ds['alive_players_ct'] = list(map(len, get_attr(ds, "CT")))
    ds['health_ct']        = list(map(sum, get_attr(ds, "CT", "health")))
    ds['health_t']         = list(map(sum, get_attr(ds, "Terrorist", "health")))
    ds['money_ct']         = list(map(sum, get_attr(ds, "CT", "money")))
    ds['money_t']          = list(map(sum, get_attr(ds, "Terrorist", "money")))
    ds['has_defuser']      = list(map(sum, get_attr(ds, "CT", "has_defuser")))
    ds['has_helmet_ct']    = list(map(sum, get_attr(ds, "CT", "has_helmet")))
    ds['has_helmet_t']     = list(map(sum, get_attr(ds, "Terrorist", "has_helmet")))
    ds['armor_ct']         = list(map(sum, get_attr(ds, "CT", "armor")))
    ds['armor_t']          = list(map(sum, get_attr(ds, "Terrorist", "armor")))
    

    
rifles = ["G3sg1", "M4a4", "GalilAr", "M4a1S", "Sg553", "Awp", "Famas" ,
          "Aug", "Ssg08", "Ak47", "Scar20"]
SMGs = ["Ump45", "Mp5sd", "Mac10",  "Mp9", "Mp7", "P90"]
heavy = ["Xm1014", "Mag7", "Nova", "Sawedoff"]
pistols = [ "Deagle", "P250", "Cz75Auto", 
           "FiveSeven", "Tec9", "Elite", "ZeusX27"]
default = ["UspS", "Glock", "P2000"]
all_weapons = rifles + SMGs + heavy + pistols + default  

def get_weapon_count(inventory_list, weapon_name):
    time_left = list(X["round_status_time_left"])
    simple_inventory = []
    index = 0
    for item in inventory_list:
        if item == "NEXT_ROW":
            index += 1                      #time startS at 114.xx next timestamp is 94.xx
        elif item["item_type"] in all_weapons and time_left[index] > 95: 
            simple_inventory.append(item["item_type"])
    #returns the number of times the item was purchased or owned at the beginning
    return simple_inventory.count(weapon_name)

    
def get_weapon_kills(weapon_name):
    weapon_kills = []
    prev_kill =  X["previous_kills"]
    for index in range(1,len(prev_kill)):
        if prev_kill[index] == [] and prev_kill[index-1] != [] :
            for weapon in prev_kill[index-1]:
                weapon_kills.append(weapon["weapon"])
    return weapon_kills.count(weapon_name), len(weapon_kills)


def get_value(weapon_name):
    if get_weapon_kills(weapon_name)[0] == 0:
        return 0
    return get_weapon_kills(weapon_name)[0] / (get_weapon_count(t_inventory, weapon_name) + get_weapon_count(ct_inventory, weapon_name))        
 

def get_firepower(weapon_name):
    return (np.mean(get_value(weapon_name) + (get_weapon_kills(weapon_name)[0] / get_weapon_kills(weapon_name)[1]))) * 100


weapon_scores_dict = dict(zip(all_weapons, [get_firepower(item) for item in all_weapons]))

def get_firepower(inventory_list):
    firepower = []
    firepower_sum = 0
    for i in inventory_list:
        if i =="NEXT_ROW":
            firepower.append(firepower_sum)
            firepower_sum = 0
        elif i["item_type"] in all_weapons:
            firepower_sum += weapon_scores_dict.get(i["item_type"])
    return firepower

X["ct_firepower"] = get_firepower(ct_inventory)
X["t_firepower"] = get_firepower(t_inventory)
X.drop(["previous_kills"],axis=1, inplace=True)
X.drop(["alive_players"], axis = 1, inplace = True) 

"""
                        2. SPLITTING AND FEATURE EXTRACTION
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#Features
columns = X_test.columns
features = ['round_status_time_left', 'round_difference',
           'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass',
           'de_train', 'de_vertigo', 'round_status_BombPlanted',
           'round_status_FreezeTime', 'round_status_Normal', 'leading_team_CT',
           'leading_team_Terrorist', 'leading_team_draw', 't_awps', 'ct_awps',
           'ct_grenades', 't_grenades', 'alive_players_t', 'alive_players_ct',
           'health_ct', 'health_t', 'money_ct', 'money_t', 'has_defuser',
           'has_helmet_ct', 'has_helmet_t', 'armor_ct', 'armor_t', "planted_on_A",
           "planted_on_B", "not_planted", "active_molotovs_count", "active_smokes_count",
           "ct_firepower", "t_firepower",'CTs_in_sector_1', 'Ts_in_sector_1',
           'CTs_in_sector_2', 'Ts_in_sector_2', 'CTs_in_sector_3',
           'Ts_in_sector_3', 'CTs_in_sector_4', 'Ts_in_sector_4',
           'CTs_in_sector_5', 'Ts_in_sector_5', 'CTs_in_sector_6',
           'Ts_in_sector_6', 'CTs_in_sector_7', 'Ts_in_sector_7',
           'CTs_in_sector_8', 'Ts_in_sector_8', 'CTs_in_sector_9']


"""
                                3. TRAINING 
"""
#Fitting model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, max_features= "sqrt", random_state=42)
model.fit(X_train[features],y_train)
    
"""
                                4. EVALUATION
"""
#Feature importance
importance = pd.DataFrame({
    'values': model.feature_importances_,
    'features': features}).sort_values(by=["values"], ascending = False) 
plt.barh(importance["features"], importance["values"])
plt.yticks(size= 7)
plt.show()       


#heatmap
data = pd.concat([X[features],y], axis = 1)
import seaborn as sns
corr_matrix = data.corr()
top_corr_features = corr_matrix.index
plt.figure(figsize=(30,30))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")



#Confusion matrix
score = model.score(X_test[features], y_test)
y_pred = model.predict(X_test[features]) #ts win
y_pred_proba = model.predict_proba(X_test[features]) #ct-t
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)


#Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(model, X[features], y, cv=20)
avg_acc=accuracies.mean()
std_acc = accuracies.std()


