import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd
pd.reset_option('all')
import gc
import polars as pl
import numpy as np
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np


CATS = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
NUMS = ['page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
        'hover_duration', 'elapsed_time_diff']
DIALOGS = ['that', 'this', 'it', 'you','find','found','Found','notebook','Wells','wells','help','need', 'Oh','Ooh','Jo', 'flag', 'can','and','is','the','to']

COLUMNS = [
    pl.col("page").cast(pl.Float32),
    (
        (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1))
        .fill_null(0)
        .clip(0, 1e9)
        .over(["session_id", "level"])
        .alias("elapsed_time_diff")
    ),
    (
        (pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1))
        .abs()
        .over(["session_id", "level"])
    ),
    (
        (pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1))
        .abs()
        .over(["session_id", "level"])
    ),
    pl.col("fqid").fill_null("fqid_None"),
    pl.col("text_fqid").fill_null("text_fqid_None")
]

name_feature = ['basic', 'undefined', 'close', 'open', 'prev', 'next']
event_name_feature = ['cutscene_click', 'person_click', 'navigate_click',
       'observation_click', 'notification_click', 'object_click',
       'object_hover', 'map_hover', 'map_click', 'checkpoint',
       'notebook_click']


sub_fqid_lists = {'0-4': ['gramps',
 'wells',
 'toentry',
 'groupconvo',
 'tomap',
 'tostacks',
 'tobasement',
 'boss',
 'cs',
 'teddy',
 'tunic.historicalsociety',
 'plaque',
 'directory',
 'tunic',
 'tunic.kohlcenter',
 'plaque.face.date',
 'notebook',
 'tunic.hub.slip',
 'tocollection',
 'tunic.capitol_0',
 'photo',
 'intro',
 'retirement_letter',
 'togrampa',
 'janitor',
 'chap1_finale',
 'report',
 'outtolunch',
 'chap1_finale_c',
 'block_0',
 'doorblock',
 'tocloset',
 'block_tomap2',
 'block_tocollection',
 'block_tomap1'],
                  '5-12': ['worker',
 'archivist',
 'gramps',
 'toentry',
 'tomap',
 'tostacks',
 'tobasement',
 'boss',
 'journals',
 'businesscards',
 'tunic.historicalsociety',
 'tofrontdesk',
 'plaque',
 'tunic.drycleaner',
 'tunic.library',
 'trigger_scarf',
 'reader',
 'directory',
 'tunic.capitol_1',
 'journals.pic_0.next',
 'tunic',
 'what_happened',
 'tunic.kohlcenter',
 'tunic.humanecology',
 'logbook',
 'businesscards.card_0.next',
 'journals.hub.topics',
 'logbook.page.bingo',
 'journals.pic_1.next',
 'reader.paper0.next',
 'trigger_coffee',
 'wellsbadge',
 'journals.pic_2.next',
 'tomicrofiche',
 'tocloset_dirty',
 'businesscards.card_bingo.bingo',
 'businesscards.card_1.next',
 'tunic.hub.slip',
 'journals.pic_2.bingo',
 'tocollection',
 'chap2_finale_c',
 'tunic.capitol_0',
 'photo',
 'reader.paper1.next',
 'businesscards.card_bingo.next',
 'reader.paper2.bingo',
 'magnify',
 'janitor',
 'tohallway',
 'outtolunch',
 'reader.paper2.next',
 'door_block_talk',
 'block_magnify',
 'reader.paper0.prev',
 'block',
 'block_0',
 'door_block_clean',
 'reader.paper2.prev',
 'reader.paper1.prev',
 'block_badge',
 'block_badge_2',
 'block_1'],
                  '13-22': ['worker',
 'gramps',
 'wells',
 'toentry',
 'confrontation',
 'crane_ranger',
 'flag_girl',
 'tomap',
 'tostacks',
 'tobasement',
 'archivist_glasses',
 'boss',
 'journals',
 'seescratches',
 'groupconvo_flag',
 'teddy',
 'expert',
 'businesscards',
 'ch3start',
 'tunic.historicalsociety',
 'tofrontdesk',
 'savedteddy',
 'plaque',
 'glasses',
 'tunic.drycleaner',
 'reader_flag',
 'tunic.library',
 'tracks',
 'tunic.capitol_2',
 'reader',
 'directory',
 'tunic.capitol_1',
 'journals.pic_0.next',
 'unlockdoor',
 'tunic',
 'tunic.kohlcenter',
 'tunic.humanecology',
 'colorbook',
 'logbook',
 'businesscards.card_0.next',
 'journals.hub.topics',
 'journals.pic_1.next',
 'journals_flag',
 'reader.paper0.next',
 'tracks.hub.deer',
 'reader_flag.paper0.next',
 'journals.pic_2.next',
 'tomicrofiche',
 'journals_flag.pic_0.bingo',
 'tocloset_dirty',
 'businesscards.card_1.next',
 'tunic.wildlife',
 'tunic.hub.slip',
 'tocage',
 'journals.pic_2.bingo',
 'tocollectionflag',
 'tocollection',
 'chap4_finale_c',
 'lockeddoor',
 'journals_flag.hub.topics',
 'reader_flag.paper2.bingo',
 'photo',
 'tunic.flaghouse',
 'reader.paper1.next',
 'directory.closeup.archivist',
 'businesscards.card_bingo.next',
 'remove_cup',
 'journals_flag.pic_0.next',
 'coffee',
 'key',
 'reader_flag.paper1.next',
 'tohallway',
 'outtolunch',
 'journals_flag.hub.topics_old',
 'journals_flag.pic_1.next',
 'reader.paper2.next',
 'reader_flag.paper2.next',
 'journals_flag.pic_1.bingo',
 'journals_flag.pic_2.next',
 'journals_flag.pic_2.bingo',
 'reader.paper0.prev',
 'reader_flag.paper0.prev',
 'reader.paper2.prev',
 'reader.paper1.prev',
 'reader_flag.paper2.prev',
 'reader_flag.paper1.prev',
 'journals_flag.pic_0_old.next',
 'journals_flag.pic_1_old.next',
 'block_nelson',
 'journals_flag.pic_2_old.next',
 'need_glasses',
 'fox'],
                 }

sub_room_lists = {'0-4': ['tunic.historicalsociety.entry',
 'tunic.historicalsociety.stacks',
 'tunic.historicalsociety.basement',
 'tunic.kohlcenter.halloffame',
 'tunic.historicalsociety.collection',
 'tunic.historicalsociety.closet',
 'tunic.capitol_0.hall'],
                  '5-12': ['tunic.historicalsociety.entry',
 'tunic.library.frontdesk',
 'tunic.historicalsociety.frontdesk',
 'tunic.historicalsociety.stacks',
 'tunic.historicalsociety.closet_dirty',
 'tunic.humanecology.frontdesk',
 'tunic.historicalsociety.basement',
 'tunic.kohlcenter.halloffame',
 'tunic.library.microfiche',
 'tunic.drycleaner.frontdesk',
 'tunic.historicalsociety.collection',
 'tunic.capitol_1.hall',
 'tunic.capitol_0.hall'],
                  '13-22': ['tunic.historicalsociety.entry',
 'tunic.wildlife.center',
 'tunic.historicalsociety.cage',
 'tunic.library.frontdesk',
 'tunic.historicalsociety.frontdesk',
 'tunic.historicalsociety.stacks',
 'tunic.historicalsociety.closet_dirty',
 'tunic.humanecology.frontdesk',
 'tunic.historicalsociety.basement',
 'tunic.kohlcenter.halloffame',
 'tunic.library.microfiche',
 'tunic.drycleaner.frontdesk',
 'tunic.historicalsociety.collection',
 'tunic.flaghouse.entry',
 'tunic.historicalsociety.collection_flag',
 'tunic.capitol_1.hall',
 'tunic.capitol_2.hall'],
                 }


sub_text_lists = {'0-4': ['tunic.historicalsociety.entry.groupconvo',
 'tunic.historicalsociety.collection.cs',
 'tunic.historicalsociety.collection.gramps.found',
 'tunic.historicalsociety.closet.gramps.intro_0_cs_0',
 'tunic.historicalsociety.closet.teddy.intro_0_cs_0',
 'tunic.historicalsociety.closet.intro',
 'tunic.historicalsociety.closet.retirement_letter.hub',
 'tunic.historicalsociety.collection.tunic.slip',
 'tunic.kohlcenter.halloffame.plaque.face.date',
 'tunic.kohlcenter.halloffame.togrampa',
 'tunic.historicalsociety.collection.gramps.lost',
 'tunic.historicalsociety.closet.notebook',
 'tunic.historicalsociety.basement.janitor',
 'tunic.historicalsociety.stacks.outtolunch',
 'tunic.historicalsociety.closet.photo',
 'tunic.historicalsociety.collection.tunic',
 'tunic.historicalsociety.closet.teddy.intro_0_cs_5',
 'tunic.historicalsociety.entry.wells.talktogramps',
 'tunic.historicalsociety.entry.boss.talktogramps',
 'tunic.historicalsociety.closet.doorblock',
 'tunic.historicalsociety.entry.block_tomap2',
 'tunic.historicalsociety.entry.block_tocollection',
 'tunic.historicalsociety.entry.block_tomap1',
 'tunic.historicalsociety.collection.gramps.look_0',
 'tunic.kohlcenter.halloffame.block_0',
 'tunic.capitol_0.hall.chap1_finale_c',
 'tunic.historicalsociety.entry.gramps.hub'],
               '5-12': ['tunic.historicalsociety.frontdesk.archivist.newspaper',
 'tunic.historicalsociety.frontdesk.archivist.have_glass',
 'tunic.drycleaner.frontdesk.worker.hub',
 'tunic.historicalsociety.closet_dirty.gramps.news',
 'tunic.humanecology.frontdesk.worker.intro',
 'tunic.library.frontdesk.worker.hello',
 'tunic.library.frontdesk.worker.wells',
 'tunic.historicalsociety.frontdesk.archivist.hello',
 'tunic.historicalsociety.closet_dirty.trigger_scarf',
 'tunic.drycleaner.frontdesk.worker.done',
 'tunic.historicalsociety.closet_dirty.what_happened',
 'tunic.historicalsociety.stacks.journals.pic_2.bingo',
 'tunic.humanecology.frontdesk.worker.badger',
 'tunic.historicalsociety.closet_dirty.trigger_coffee',
 'tunic.drycleaner.frontdesk.logbook.page.bingo',
 'tunic.library.microfiche.reader.paper2.bingo',
 'tunic.historicalsociety.closet_dirty.gramps.helpclean',
 'tunic.historicalsociety.frontdesk.archivist.have_glass_recap',
 'tunic.historicalsociety.frontdesk.magnify',
 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo',
 'tunic.library.frontdesk.wellsbadge.hub',
 'tunic.capitol_1.hall.boss.haveyougotit',
 'tunic.historicalsociety.basement.janitor',
 'tunic.historicalsociety.closet_dirty.photo',
 'tunic.historicalsociety.stacks.outtolunch',
 'tunic.library.frontdesk.worker.wells_recap',
 'tunic.capitol_0.hall.boss.talktogramps',
 'tunic.historicalsociety.closet_dirty.gramps.archivist',
 'tunic.historicalsociety.closet_dirty.door_block_talk',
 'tunic.historicalsociety.frontdesk.archivist.need_glass_0',
 'tunic.historicalsociety.frontdesk.block_magnify',
 'tunic.historicalsociety.frontdesk.archivist.foundtheodora',
 'tunic.historicalsociety.closet_dirty.gramps.nothing',
 'tunic.historicalsociety.closet_dirty.door_block_clean',
 'tunic.library.frontdesk.worker.hello_short',
 'tunic.historicalsociety.stacks.block',
 'tunic.historicalsociety.frontdesk.archivist.need_glass_1',
 'tunic.historicalsociety.frontdesk.archivist.newspaper_recap',
 'tunic.drycleaner.frontdesk.worker.done2',
 'tunic.humanecology.frontdesk.block_0',
 'tunic.library.frontdesk.worker.preflag',
 'tunic.drycleaner.frontdesk.worker.takealook',
 'tunic.library.frontdesk.worker.droppedbadge',
 'tunic.library.microfiche.block_0',
 'tunic.library.frontdesk.block_badge',
 'tunic.library.frontdesk.block_badge_2',
 'tunic.capitol_1.hall.chap2_finale_c',
 'tunic.drycleaner.frontdesk.block_0',
 'tunic.humanecology.frontdesk.block_1',
 'tunic.drycleaner.frontdesk.block_1'],
               '13-22': ['tunic.historicalsociety.cage.confrontation',
 'tunic.wildlife.center.crane_ranger.crane',
 'tunic.wildlife.center.wells.nodeer',
 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation',
 'tunic.historicalsociety.basement.seescratches',
 'tunic.flaghouse.entry.flag_girl.hello',
 'tunic.historicalsociety.basement.ch3start',
 'tunic.historicalsociety.entry.groupconvo_flag',
 'tunic.historicalsociety.collection_flag.gramps.flag',
 'tunic.historicalsociety.basement.savedteddy',
 'tunic.library.frontdesk.worker.nelson',
 'tunic.wildlife.center.expert.removed_cup',
 'tunic.library.frontdesk.worker.flag',
 'tunic.historicalsociety.entry.boss.flag',
 'tunic.flaghouse.entry.flag_girl.symbol',
 'tunic.wildlife.center.wells.animals',
 'tunic.historicalsociety.cage.glasses.afterteddy',
 'tunic.historicalsociety.cage.teddy.trapped',
 'tunic.historicalsociety.cage.unlockdoor',
 'tunic.historicalsociety.stacks.journals.pic_2.bingo',
 'tunic.historicalsociety.entry.wells.flag',
 'tunic.humanecology.frontdesk.worker.badger',
 'tunic.historicalsociety.stacks.journals_flag.pic_0.bingo',
 'tunic.historicalsociety.entry.directory.closeup.archivist',
 'tunic.capitol_2.hall.boss.haveyougotit',
 'tunic.wildlife.center.wells.nodeer_recap',
 'tunic.historicalsociety.cage.glasses.beforeteddy',
 'tunic.wildlife.center.expert.recap',
 'tunic.historicalsociety.stacks.journals_flag.pic_1.bingo',
 'tunic.historicalsociety.cage.lockeddoor',
 'tunic.historicalsociety.stacks.journals_flag.pic_2.bingo',
 'tunic.wildlife.center.remove_cup',
 'tunic.wildlife.center.tracks.hub.deer',
 'tunic.historicalsociety.frontdesk.key',
 'tunic.library.microfiche.reader_flag.paper2.bingo',
 'tunic.flaghouse.entry.colorbook',
 'tunic.wildlife.center.coffee',
 'tunic.historicalsociety.collection_flag.gramps.recap',
 'tunic.wildlife.center.wells.animals2',
 'tunic.flaghouse.entry.flag_girl.symbol_recap',
 'tunic.historicalsociety.closet_dirty.photo',
 'tunic.historicalsociety.stacks.outtolunch',
 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation_recap',
 'tunic.historicalsociety.entry.boss.flag_recap',
 'tunic.capitol_1.hall.boss.writeitup',
 'tunic.library.frontdesk.worker.nelson_recap',
 'tunic.historicalsociety.entry.wells.flag_recap',
 'tunic.drycleaner.frontdesk.worker.done2',
 'tunic.library.frontdesk.worker.flag_recap',
 'tunic.library.frontdesk.worker.preflag',
 'tunic.historicalsociety.basement.gramps.seeyalater',
 'tunic.flaghouse.entry.flag_girl.hello_recap',
 'tunic.historicalsociety.basement.gramps.whatdo',
 'tunic.library.frontdesk.block_nelson',
 'tunic.historicalsociety.cage.need_glasses',
 'tunic.capitol_2.hall.chap4_finale_c',
 'tunic.wildlife.center.fox.concern']
              }


SUB_LEVELS = {'0-4': [1, 2, 3, 4],
              '5-12': [5, 6, 7, 8, 9, 10, 11, 12],
              '13-22': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]}
level_groups = ["0-4", "5-12", "13-22"]


def feature_engineer(x, grp, use_extra, feature_suffix):
    LEVELS = SUB_LEVELS[grp]
    text_lists = sub_text_lists[grp]
    room_lists = sub_room_lists[grp]
    fqid_lists = sub_fqid_lists[grp]
    aggs = [
        pl.col("index").count().alias(f"session_number_{feature_suffix}"),

        *[pl.col('index').filter(pl.col('text').str.contains(c)).count().alias(f'word_{c}') for c in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).mean().alias(f'word_mean_{c}') for c in
          DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).std().alias(f'word_std_{c}') for c in
          DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).max().alias(f'word_max_{c}') for c in
          DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).min().alias(f'word_min_{c}') for c in
          DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).sum().alias(f'word_sum_{c}') for c in
          DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).median().alias(f'word_median_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).skew().alias(f'word_skew_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).kurtosis().alias(f'word_kurtosis_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).quantile(0.25).alias(f'word_quant25_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).quantile(0.3).alias(f'word_quant3_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).quantile(0.8).alias(f'word_quant8_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).quantile(0.5).alias(f'word_quant5_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).quantile(0.65).alias(f'word_quant65_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).quantile(0.75).alias(f'word_quant75_{c}') for c
          in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).var().alias(f'word_var_{c}') for c
          in DIALOGS],
        *[(pl.col("elapsed_time_diff").filter(pl.col("text") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("text") == c).min()).alias(f"{c}_word_max_min_{feature_suffix}") for c in DIALOGS],



        *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in CATS],

        *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in NUMS],
        *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in NUMS],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in NUMS],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in NUMS],
        *[pl.col(c).median().alias(f"{c}_median_{feature_suffix}") for c in NUMS],
        *[pl.col(c).skew().alias(f"{c}_skew_{feature_suffix}") for c in NUMS],
        *[pl.col(c).kurtosis().alias(f"{c}_kurtosis_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.25).alias(f"{c}_quant25_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.75).alias(f"{c}_quant75_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.3).alias(f"{c}_quant3_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.8).alias(f"{c}_quant8_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.5).alias(f"{c}_quant5_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.65).alias(f"{c}_quant65_{feature_suffix}") for c in NUMS],
        *[pl.col(c).var().alias(f"{c}_var_{feature_suffix}") for c in NUMS],
        *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_max_min_{feature_suffix}") for c in NUMS],
 



        *[pl.col("fqid").filter(pl.col("fqid") == c).count().alias(f"{c}_fqid_counts{feature_suffix}")
          for c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for 
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for 
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).var().alias(f"{c}_ET_var_{feature_suffix}") for
          c in fqid_lists],
        *[(pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("fqid") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in fqid_lists],




        *[pl.col("text_fqid").filter(pl.col("text_fqid") == c).count().alias(f"{c}_text_fqid_counts{feature_suffix}")
          for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}")
          for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for
          c in text_lists],
         *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).var().alias(f"{c}_ET_var_{feature_suffix}") for
          c in text_lists],
        *[(pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("text_fqid") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in text_lists],


        *[pl.col("room_fqid").filter(pl.col("room_fqid") == c).count().alias(f"{c}_room_fqid_counts{feature_suffix}")
          for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in room_lists], 
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).var().alias(f"{c}_ET_var_{feature_suffix}") for
          c in room_lists],
        *[(pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("room_fqid") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in room_lists],


        *[pl.col("event_name").filter(pl.col("event_name") == c).count().alias(f"{c}_event_name_counts{feature_suffix}")
          for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
          for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).median().alias(
            f"{c}_ET_median_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).var().alias(f"{c}_ET_var_{feature_suffix}") for
          c in event_name_feature],
        *[(pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("event_name") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in event_name_feature],

        *[pl.col("name").filter(pl.col("name") == c).count().alias(f"{c}_name_counts{feature_suffix}") for c in
          name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in
          name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in
          name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in
          name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for
          c in
          name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for
          c in name_feature],
         *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in name_feature], 
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).var().alias(f"{c}_ET_var_{feature_suffix}") for
          c in name_feature],
        *[(pl.col("elapsed_time_diff").filter(pl.col("name") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("name") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in name_feature],

        *[pl.col("level").filter(pl.col("level") == c).count().alias(f"{c}_LEVEL_count{feature_suffix}") for c in
          LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in
          LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c
          in
          LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in
          LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for
          c in
          LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).var().alias(f"{c}_var_{feature_suffix}") for
          c in LEVELS],
        *[(pl.col("elapsed_time_diff").filter(pl.col("level") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("level") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in LEVELS],


        *[pl.col("level_group").filter(pl.col("level_group") == c).count().alias(
            f"{c}_LEVEL_group_count{feature_suffix}") for c in
          level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for
          c in
          level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}")
          for c in
          level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for
          c in
          level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).median().alias(
            f"{c}_ET_median_{feature_suffix}") for c in
          level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).skew().alias(f"{c}_ET_skew_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).kurtosis().alias(f"{c}_ET_kurtosis_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).quantile(0.25).alias(f"{c}_ET_quant25_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).quantile(0.75).alias(f"{c}_ET_quant75_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).quantile(0.3).alias(f"{c}_ET_quant3_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).quantile(0.8).alias(f"{c}_ET_quant8_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).quantile(0.5).alias(f"{c}_ET_quant5_{feature_suffix}") for
          c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).quantile(0.65).alias(f"{c}_ET_quant65_{feature_suffix}") for
          c in level_groups],
        *[(pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).max() - pl.col("elapsed_time_diff").filter(
            pl.col("level_group") == c).min()).alias(f"{c}_ET_max_min_{feature_suffix}") for c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).var().alias(f"{c}_ET_var_{feature_suffix}") for
          c in level_groups]]

    # df = x.groupby(['session_id']).agg(aggs).sort_values("session_id")

    df = x.with_columns(COLUMNS).groupby(['session_id'], maintain_order=True).agg(aggs).sort("session_id")

    if use_extra:
        if grp == '5-12':
            aggs = [
                pl.col("elapsed_time").filter((pl.col("text") == "Here's the log book.")
                                              | (pl.col("fqid") == 'logbook.page.bingo'))
                    .apply(lambda s: s.max() - s.min()).alias("logbook_bingo_duration"),
                pl.col("index").filter(
                    (pl.col("text") == "Here's the log book.") | (pl.col("fqid") == 'logbook.page.bingo')).apply(
                    lambda s: s.max() - s.min()).alias("logbook_bingo_indexCount"),
                pl.col("elapsed_time").filter(
                    ((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'reader')) | (
                            pl.col("fqid") == "reader.paper2.bingo")).apply(lambda s: s.max() - s.min()).alias(
                    "reader_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'reader')) | (
                        pl.col("fqid") == "reader.paper2.bingo")).apply(lambda s: s.max() - s.min()).alias(
                    "reader_bingo_indexCount"),
                pl.col("elapsed_time").filter(
                    ((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'journals')) | (
                            pl.col("fqid") == "journals.pic_2.bingo")).apply(lambda s: s.max() - s.min()).alias(
                    "journals_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'journals')) | (
                        pl.col("fqid") == "journals.pic_2.bingo")).apply(lambda s: s.max() - s.min()).alias(
                    "journals_bingo_indexCount"),
            ]
            #tmp = x.groupby(['session_id']).agg(aggs).sort_values("session_id")

            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
            df = df.join(tmp, on="session_id", how='left')

        if grp == '13-22':
            aggs = [
                pl.col("elapsed_time").filter(
                    ((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'reader_flag')) | (
                            pl.col("fqid") == "tunic.library.microfiche.reader_flag.paper2.bingo")).apply(
                    lambda s: s.max() - s.min() if s.len() > 0 else 0).alias("reader_flag_duration"),
                pl.col("index").filter(
                    ((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'reader_flag')) | (
                            pl.col("fqid") == "tunic.library.microfiche.reader_flag.paper2.bingo")).apply(
                    lambda s: s.max() - s.min() if s.len() > 0 else 0).alias("reader_flag_indexCount"),
                pl.col("elapsed_time").filter(
                    ((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'journals_flag')) | (
                            pl.col("fqid") == "journals_flag.pic_0.bingo")).apply(
                    lambda s: s.max() - s.min() if s.len() > 0 else 0).alias("journalsFlag_bingo_duration"),
                pl.col("index").filter(
                    ((pl.col("event_name") == 'navigate_click') & (pl.col("fqid") == 'journals_flag')) | (
                            pl.col("fqid") == "journals_flag.pic_0.bingo")).apply(
                    lambda s: s.max() - s.min() if s.len() > 0 else 0).alias("journalsFlag_bingo_indexCount")
            ]
            #tmp = x.groupby(['session_id']).agg(aggs).sort_values("session_id")

            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
            df = df.join(tmp, on="session_id", how='left')

    return df.to_pandas()

def time_feature(train):
    train = train.reset_index()
    q = (
        pl.from_pandas(train)
          .lazy()
          .with_columns([
            pl.col("session_id").apply(lambda x: int(str(x)[:2])).alias('year'),#.astype(np.uint8)
            pl.col("session_id").apply(lambda x: int(str(x)[2:4])+1).alias('month'),#.astype(np.uint8)
            pl.col("session_id").apply(lambda x: int(str(x)[4:6])).alias('day'),#.astype(np.uint8)
            pl.col("session_id").apply(lambda x: int(str(x)[6:8])).alias('hour'),#.astype(np.uint8)
            pl.col("session_id").apply(lambda x: int(str(x)[8:10])).alias('minute'),#.astype(np.uint8)
            pl.col("session_id").apply(lambda x: int(str(x)[10:12])).alias('second'),#.astype(np.uint8)
            pl.col("session_id").apply(lambda x: int(str(x)[12:])).alias('id_anonymous'),#.astype(np.uint8)
        ])
    )

    return q.collect().to_pandas().set_index('session_id')


# https://www.kaggle.com/code/glipko/recap-texts#Data-Extraction-
def text_cnt(x, revised_train):
    
    x['in_the_same_dialogue'] = x['text_fqid'].shift() # 한칸씩 뒤로
    x['in_the_same_dialogue'] = x['text_fqid'] == x['in_the_same_dialogue']

    dialogue_sequence = x[~x['in_the_same_dialogue']] # 겹치지 않는 아이들
    dialogue_sequence = dialogue_sequence[~dialogue_sequence['text_fqid'].isna()]
    dialogue_sequence = dialogue_sequence[(dialogue_sequence['event_name'] == 'observation_click') | \
                                      (dialogue_sequence['event_name'] == 'person_click')]
    dialogue_sequence = dialogue_sequence.drop(columns=['text', 'in_the_same_dialogue', 'elapsed_time'], errors='ignore')

    dialogues = x[x['event_name'] == 'person_click']['text_fqid'].unique()
    recap_dialogues = []
    for dialogue in dialogues:
        if('recap' in dialogue or 'lost' in dialogue):
            recap_dialogues.append(dialogue)


    observations = x[x['event_name'] == 'observation_click']['text_fqid'].unique()
    recap_observations = []
    for observation in observations:
        if('block' in observation):
            recap_observations.append(observation)


    dialogue_sequence = dialogue_sequence[dialogue_sequence['text_fqid'].isin(recap_observations) | \
                        dialogue_sequence['text_fqid'].isin(recap_dialogues)]

    session_event_recap = dialogue_sequence.groupby(['session_id', 'event_name']) \
                                        .size() \
                                        .reset_index() \
                                        .rename(columns={0:'recap_reading'})

    session_event_recap = session_event_recap[(session_event_recap['event_name'] == 'observation_click') | \
                    (session_event_recap['event_name'] == 'person_click')]

    session_recap = session_event_recap.groupby('session_id')['recap_reading'].sum()

    texts = x[(~x['text'].isna()) & (x['text'] != 'undefined')]
    reading = texts.groupby('session_id').size()
    text_feature=pd.concat([session_recap,reading], axis=1)
    text_feature.columns=['recap_reading', 'reading_cnt']
    
    # revised_train = pd.merge(x, revised_train, on='session_id', how='left')
    
    return pd.merge(revised_train, text_feature, on='session_id', how='left').set_index('session_id')


def feature_quest(new_train, train, q):
    train_q = new_train.copy()
    texts = {
        1: ["Yes! This cool old slip from 1916.", 
             "Go ahead, take a peek at the shirt!", 
             "I'll be at the Capitol. Let me know if you find anything!", 
             "We need to talk about that missing paperwork.", 
             "The slip is from 1916 but the team didn't start until 1974!"], 
         2: ["It's already all done!", 
             "Gramps is the best historian ever!"], 
         3: ["I suppose historians are boring, too?" 
             "Why don't you head to the Basketball Center and rustle up some clues?", 
             "We need to talk about that missing paperwork."],    
        
         4: ['I need to find the owner of this slip.',
             'She led marches and helped women get the right to vote!', 
             "Here's a call number to find more info in the Stacks.", 
             "What was Wells doing here?"],

         5: ["Your gramps is awesome! Always full of stories.",
             "Here's a call number to find more info in the Stacks.", 
             "Where did you get that coffee?"],         
        
         6: ["Oh, that's from Bean Town.", 
             "Wells? I knew it!"], 
           
         7: ["Try not to panic, Jo.",
             "I've got a stack of business cards from my favorite cleaners.",
             "Check out our microfiche. It's right through that door.", 
             "I'm afraid my papers have gone missing in this mess.", 
             "Nope. But Youmans and other suffragists worked hard to change that."], 
            
         8: ["What should I do first?",
             "Thanks to them, Wisconsin was the first state to approve votes for women!"], 

         9: [ "Can you help me? I need to find the owner of this slip.",
             'Looks like a dry cleaning receipt.',
             "I knew I could count on you, Jo!", 
             "Nope, that's from Bean Town. I only drink Holdgers!"], 

         10:["I love these photos of me and Teddy."
             'Your gramps is awesome! Always full of stories.',
             "Nope. But Youmans and other suffragists worked hard to change that.", 
             "Right outside the door.", 
             "Do you have any info on Theodora Youmans?"], 
                   
         11:["I ran into Wells there this morning",
             'Your gramps is awesome! Always full of stories.',
             "Wait a sec. Women couldn't vote?!", 
             "I've got a stack of business cards from my favorite cleaners.",
             "An old shirt? Try the university."],  
         12:[],
         13:[],        
         14:[],
         15:[],
         16:[],
         17:[],
         18:[]
        }
    i = 0
    for text in texts[q]:
        i += 1
        train_q['text' + str(i)] = train[train['text'] == text].groupby(['session_id'])['delt_time'].sum()
    
    fqids = {
         1: ['directory'], 
         2: ['notebook','chap1_finale_c'], 
         3: ['tostacks','doorblock'], 
         4: ['journals.pic_1.next', 'businesscards.card_1.next', 'block'], 
         5: ['janitor', 'journals.pic_2.next'], 
         6: ['businesscards', 'journals.pic_0.next','tobasement', 'logbook.page.bingo', 'tohallway'],  
         7: ['journals.pic_1.next','reader.paper2.bingo','businesscards.card_bingo.next', 
             'logbook.page.bingo', 'tunic.kohlcenter'],  
         8: ['reader.paper2.bingo'],  
         9: ['journals.pic_1.next','businesscards.card_bingo.bingo', 'reader'],  
         10:['tunic.kohlcenter','magnify','block','journals.pic_1.next', 'journals'], 
         11:['tostacks','block_magnify','block','businesscards.card_bingo.next'], 
         12:['businesscards.card_1.next','tofrontdesk'],  
         13:['tocloset_dirty','reader.paper1.next'], 
         14:['tracks'], 
         15:['groupconvo_flag'], 
         16:['savedteddy'], 
         17:['journals_flag.pic_0.next'], 
         18:['chap4_finale_c'], 
        }
    for fqid in fqids[q]:
        train_q['t_fqid_' + fqid] = train[train['fqid'] == fqid].groupby(['session_id'])['delt_time'].sum()

    text_fqids = {
        1:[],
        2:['tunic.historicalsociety.collection.gramps.found'],
        3:[],
        4: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.library.frontdesk.worker.wells', 
            'tunic.library.frontdesk.worker.hello'], 
        5: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.historicalsociety.closet_dirty.gramps.helpclean',
            'tunic.historicalsociety.closet_dirty.gramps.news'],     
        6: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.historicalsociety.frontdesk.archivist.foundtheodora',
            'tunic.historicalsociety.closet_dirty.trigger_coffee', 
            'tunic.historicalsociety.closet_dirty.gramps.archivist'], 
        7: ['tunic.historicalsociety.closet_dirty.door_block_talk',
            'tunic.drycleaner.frontdesk.worker.hub',
            'tunic.historicalsociety.closet_dirty.trigger_coffee'], 
        8: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.historicalsociety.frontdesk.magnify', 
            'tunic.historicalsociety.closet_dirty.trigger_coffee'], 
        9: ['tunic.historicalsociety.frontdesk.archivist.hello',
            'tunic.library.frontdesk.worker.wells', 
            'tunic.historicalsociety.frontdesk.archivist.foundtheodora'], 
        10: ['tunic.library.frontdesk.worker.wells',
            'tunic.historicalsociety.frontdesk.archivist.have_glass_recap',
             'tunic.historicalsociety.closet_dirty.gramps.news'], 
        11: ['tunic.historicalsociety.frontdesk.archivist.newspaper_recap',
             'tunic.historicalsociety.closet_dirty.gramps.archivist'], 
        12:[],
        13:['tunic.drycleaner.frontdesk.logbook.page.bingo'],
        14: ['tunic.flaghouse.entry.flag_girl.symbol_recap', 
             'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation_recap'],
        15:['tunic.flaghouse.entry.colorbook'], 
        16:['tunic.library.frontdesk.worker.nelson'], 
        17:['tunic.historicalsociety.entry.wells.flag'], 
        18:['tunic.flaghouse.entry.flag_girl.symbol_recap'], 
    }
    for text_fqid in text_fqids[q]:
        maska = train['text_fqid'] == text_fqid
        train_q['t_text_fqid_' + text_fqid] = train[maska].groupby(['session_id'])['delt_time'].sum()       
        train_q['l_text_fqid_' + text_fqid] = train[train['text_fqid'] == text_fqid].groupby(['session_id'])['index'].count()


    room_lvls = {
         1: [['tunic.capitol_0.hall',4],['tunic.historicalsociety.collection',3],
            ['tunic.historicalsociety.entry',1],['tunic.historicalsociety.collection', 2]], 
         2: [],
         3: [['tunic.capitol_0.hall',4]], 
         4: [['tunic.historicalsociety.frontdesk',12], 
             ['tunic.historicalsociety.stacks',7]], 
         5: [['tunic.historicalsociety.stacks',12]],  
         6: [['tunic.drycleaner.frontdesk',8],  
             ['tunic.library.microfiche',9]], 
         7: [['tunic.library.frontdesk',10]], 
         8: [['tunic.kohlcenter.halloffame', 11], 
             ['tunic.kohlcenter.halloffame',6]], 
         9: [['tunic.capitol_1.hall', 12], 
             ['tunic.historicalsociety.collection',12]],
         10:[['tunic.humanecology.frontdesk',7]], 
         11:[['tunic.drycleaner.frontdesk',9], 
             ['tunic.historicalsociety.collection',6]], 
         12:[['tunic.historicalsociety.stacks',6],
             ['tunic.historicalsociety.frontdesk', 7],
             ['tunic.historicalsociety.closet_dirty',11], 
             ['tunic.historicalsociety.frontdesk', 12]], 
         13:[['tunic.library.microfiche', 9], 
             ['tunic.historicalsociety.stacks', 11],
             ['tunic.library.frontdesk', 10], 
             ['tunic.historicalsociety.entry', 5]], 
         14:[['tunic.historicalsociety.closet_dirty',17],
             ['tunic.historicalsociety.entry',15]], 
         15:[['tunic.historicalsociety.entry',15],
             ['tunic.library.frontdesk',20]], 
         16:[['tunic.library.frontdesk', 20],
             ['tunic.wildlife.center',19]], 
         17:[['tunic.wildlife.center', 19],
             ['tunic.historicalsociety.stacks', 21]], 
         18:[['tunic.wildlife.center', 22]], 
        }
    for rl in room_lvls[q]:
        nam = rl[0]+str(rl[1])
        maska = (train['room_fqid'] == rl[0])&(train['level'] == rl[1])
        train_q['t_' + nam] = train[maska].groupby(['session_id'])['delt_time'].sum()
        train_q['l_' + nam] = train[maska].groupby(['session_id'])['index'].count()

    return train_q

def load_targets(args):
    targets = pd.read_csv(args.target)
    targets["session"] = targets["session_id"].str.split("_",expand = True)[0]
    targets["session"] = targets["session"].astype(int)
    targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
    return targets

def delt_time_def(df):
    df.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    df['elapsed_time_diff'] = df['elapsed_time'].diff(1)
    df['elapsed_time_diff'].fillna(0, inplace=True)
    df['delt_time'] = df['elapsed_time_diff'].clip(0, 103000)  
    df['delt_time_next'] = df['delt_time'].shift(-1)
    return df


# 수정
def new_page(X, revised_train, grp):
    '''
    X = original train dataset
    '''
    # 이상치 session_id 추출
    if grp == '5-12':
        session_2 = X.loc[X.page == 0, 'session_id'].unique()
        X['new_page'] = 1
        X.loc[X.session_id.isin(session_2), 'new_page'] = 0

        # new_page == 0이라면
        page_0 = X[X['session_id'].isin(session_2)]
        kur0 = page_0.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.kurtosis()).fillna(0)
        skew0 = page_0.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.skew()).fillna(0)

        pd_kur0 = pd.DataFrame({'new_page_kur': kur0})
        pd_sk0 = pd.DataFrame({'new_page_skew': skew0})

        col_total0 = pd.concat([pd_kur0, pd_sk0], axis=1)

        # new_page == 1이라면
        page_1 = X[~X['session_id'].isin(session_2)]
        kur = page_1.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.kurtosis()).fillna(0)
        skew = page_1.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.skew()).fillna(0)

        pd_kur = pd.DataFrame({'new_page_kur': kur})
        pd_sk = pd.DataFrame({'new_page_skew': skew})

        col_total1 = pd.concat([pd_kur, pd_sk], axis=1)

        total = pd.concat([col_total0, col_total1], axis=0)

        del session_2, page_0, kur0, skew0, pd_kur0, pd_sk0, page_1, kur, skew, pd_kur, pd_sk, col_total0, col_total1
        gc.collect()

        return pd.merge(revised_train, total, left_index=True, right_index=True, how='left')
    
    elif grp == '13-22':
        session_3 = X.loc[X.page.isin([0, 1, 2]), 'session_id'].unique()
        X['new_page'] = 1
        X.loc[X.session_id.isin(session_3), 'new_page'] = 0

        # new_page == 0이라면
        page_0 = X[X['session_id'].isin(session_3)]
        kur0 = page_0.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.kurtosis()).fillna(0)
        skew0 = page_0.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.skew()).fillna(0)

        pd_kur0 = pd.DataFrame({'new_page_kur': kur0})
        pd_sk0 = pd.DataFrame({'new_page_skew': skew0})

        col_total0 = pd.concat([pd_kur0, pd_sk0], axis=1)

        # new_page == 1이라면
        page_1 = X[~X['session_id'].isin(session_3)]
        kur = page_1.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.kurtosis()).fillna(0)
        skew = page_1.groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.skew()).fillna(0)

        pd_kur = pd.DataFrame({'new_page_kur': kur})
        pd_sk = pd.DataFrame({'new_page_skew': skew})

        col_total1 = pd.concat([pd_kur, pd_sk], axis=1)

        total = pd.concat([col_total0, col_total1], axis=0)

        del session_3, page_0, kur0, skew0, pd_kur0, pd_sk0, page_1, kur, skew, pd_kur, pd_sk, col_total0, col_total1
        gc.collect()

        return pd.merge(revised_train, total, left_index=True, right_index=True, how='left')
    else:
        return revised_train



# group별 tfidf기준 correct에서 상대적으로 중요하게 나온 단어
group1_text_corr=['thing', 'fascinating', 'yes', 'been', 'he', 'wow', 'done', 'good', 'closer', 'shouldn', 'enough', 'plus', 'great', 'ugh', 'collection', 'waiting', 'room', 'catch'] 
group2_text_corr=['cough', 'like', 'grumble', 'little', 'ooh', 'mission', 'time', 'horses', 'hold', 'wayyyy', 'while', 'yikes', 'funny', 'yup', 'sore', 'kid', 'letters', 'tiny', 'pony', 'missions', 'throat', 'blasted', 'two', 'actually', 'horse', 'or', 'hoarse', 'huh', 'ransacked', 'doesn', 'been', 'then', 'kidnapped', 'really', 'drop', 'something', 'dear', 'card', 'floor'] 
group1_text_wron=['undefined', 'look', 'did', 'clues', 'back', 'of', 'best', 'later', 'grampa', 'them', 'ooh', 'bee', 'knees', 'love', 'photos', 'these', 'button', 'ever', 'better', 'with', 'check', 'hopefully', 'um', 'want', 'again', 'because', 'couldn', 'those', 'hooray', 'never', 'works', 'play', 'should', 'feel', 'forgetting', 'something', 'around']
group2_text_wron=['on', 'archivist', 'should', 'go', 'out', 'stacks', 'haven', 'any', 'talk', 'university', 'well', 'figure', 'cleaning', 'so', 'capitol', 'wearing', 'check', 'better', 'upstairs', 'badgers', 'undefined', 'still', 'work', 'love', 'these', 'photos', 'button', 'jolie', 'poor', 'make', 'hear', 'loose', 'museum', 'lost', 'glasses', 'believe', 'reading', 'yeah', 'head', 'again', 'along', 'run', 'counting', 'yet', 'quite', 'figured', 'hoping', 'glass', 'u00e2', 'by', 'used', 'magnifying', 'news', 'u20ac', 'stop', 'u00a6', 'artifact', 'much', 'working', 'hmm', 'works', 'over', 'waiting', 'clue', 'clean', 'might', 'able', 'said', 'somewhere', 'weren', 'going', 'later', 'nice', 'some', 'libarian', 'information', 'seeing', 'decorations', 'such', 'day', 'fall']
group3_text_wron=['any', 'artifact', 'by', 'if', 'right', 'digging', 'were', 'library', 'well', 'figured', 'quite', 'figure', 'stacks', 'able', 'badgers', 'friend', 'outside', 'haven', 'hoping', 'stop', 'news', 'yet', 'counting', 'door', 'they', 'funny', 'lynx', 'info', 'book', 'anyway', 'newspapers', 'loaded', 'slowing', 'fall', 'such', 'later', 'seeing', 'luck', 'ready', 'when', 'hear', 'loose', 'wearing', 'shirt', 'youmans']


def add_text(data, revised_data, col, word_list, col_name_mean, col_name_std, col_name_max, col_name_kurtosis, col_name_skew):
    '''
    data= data set ex. df1, df2, df3
    col= 수정하고 싶은 열이름 str
    word_list = tfidf relative_diff (word_list)
    col_name: 새로 만들고 싶은 열이름 str
    '''
    # Create a boolean mask for rows containing the specific words
    word_mask = data[col].str.contains('|'.join(word_list), case=False)
    word_mask.fillna(False, inplace=True)

    # Group the data by 'session_id' and calculate the mean of 'time_diff' for the selected rows
    result1=data[word_mask].groupby('session_id')['elapsed_time_diff'].mean()
    result1.reindex(data['session_id'], fill_value=0)
    result2=data[word_mask].groupby('session_id')['elapsed_time_diff'].std()
    result2.reindex(data['session_id'], fill_value=0)
    result3=data[word_mask].groupby('session_id')['elapsed_time_diff'].max()
    result3.reindex(data['session_id'], fill_value=0)
    result4 = data[word_mask].groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.kurtosis()).fillna(0)
    result4.reindex(data['session_id'], fill_value=0)
    result5 = data[word_mask].groupby('session_id')['elapsed_time_diff'].apply(lambda x: x.skew()).fillna(0)
    result5.reindex(data['session_id'], fill_value=0)


    mean_=pd.DataFrame({col_name_mean:result1})
    std_=pd.DataFrame({col_name_std:result2})
    max_=pd.DataFrame({col_name_max:result3})
    kur_=pd.DataFrame({col_name_kurtosis:result4})
    ske_=pd.DataFrame({col_name_skew:result5})

    total= pd.concat([mean_,std_, max_,kur_,ske_], axis=1)

    return pd.merge(revised_data,total, left_index=True, right_index=True, how='left')


def fichi_1_2_3_4_python(train_row,revised_train, text_replace_dic, script_version_dic): 
    #print("#1 'script_version'")
    #1 'script_version'
    train_row_temp = train_row[['session_id', 'index', 'text' , 'level_group'  ]].copy()
    train_row_temp = train_row_temp.merge(script_version_dic, on='text')

    train_row_temp = train_row_temp[train_row_temp[['script_version']].notnull().all(1)]
    train_row_temp = train_row_temp.groupby(['session_id','script_version'])['session_id'].count().reset_index(name="cnt")

    train_row_temp['num_row'] = (train_row_temp
      .sort_values(by=['cnt'],
                   ascending=False) 
      .groupby('session_id', sort=False).cumcount().add(1)
    )
    train_row_temp = train_row_temp.loc[train_row_temp.num_row == 1] 
    train_row_temp = train_row_temp[['session_id','script_version']]
    train_row = train_row.merge(train_row_temp, on='session_id')
    del train_row_temp
    gc.collect()

    #print("#2 3 'delta_index_0_4'   'delta_index_5_12'")
    #2 3 'delta_index_0_4'   'delta_index_5_12'
    tmp = train_row[['session_id','index','level_group']]
    f = lambda x: x.diff()

    tmp['delta_index'] = tmp.sort_values(['session_id','index']).groupby(['session_id'])['index'].transform(f)
    tmp = tmp[tmp['delta_index'] >1]


    tmp5_12  = tmp[tmp['level_group'] == '5-12' ].groupby( ['session_id'   ]).agg({'delta_index':'max'}) .copy() 
    tmp5_12  = tmp5_12.rename(columns={"delta_index": "delta_index_0_4" })

    tmp5_12_1=tmp5_12.copy()
    tmp5_12_2=tmp5_12.copy()    
    tmp5_12_1["level_group"]='5-12'
    tmp5_12_2["level_group"]='13-22'    
    tmp5_12 = tmp5_12_1.append(tmp5_12_2)      
    train_row = train_row.merge( tmp5_12, how = 'left', left_on=[ 'session_id', "level_group"] , right_on=[ 'session_id',"level_group"] ).copy()    

    del  tmp5_12, tmp5_12_1, tmp5_12_2 
    gc.collect()    

    tmp13_22  = tmp[tmp['level_group'] == '13-22' ].groupby( ['session_id'   ]).agg({'delta_index':'max'}) .copy()      
    tmp13_22 = tmp13_22.rename(columns={"delta_index": "delta_index_5_12" })


    tmp13_22["level_group"]='13-22'   
    train_row = train_row.merge( tmp13_22, how = 'left', left_on=[ 'session_id',"level_group"] , right_on=['session_id', "level_group"] ).copy() 


    del tmp, tmp13_22
    gc.collect()

    #print("# 4 'text_replace'")
    # 4 'text_replace'
    train_row = pd.merge(train_row, text_replace_dic, how = 'left', left_on=["text"] , right_on=["text"] ).copy()
    train_row["text_replace"] = train_row["text_replace"].combine_first( train_row["text"])    

    train_row[[ 'delta_index_0_4', 'delta_index_5_12']] = train_row[[ 'delta_index_0_4', 'delta_index_5_12']].fillna(0)
    train_row[[ 'script_version']] = train_row[[ 'script_version']].fillna('') 
    train_row=pd.get_dummies(train_row, columns=['script_version'],dummy_na=False)
    
    version=train_row.groupby(['session_id'])['script_version_original', 'script_version_nohumor','script_version_nosnark', 'script_version_dry'].agg('sum')
    delta=train_row.groupby(['session_id'])['delta_index_0_4','delta_index_5_12'].agg('mean')
    
    total= pd.concat([version, delta], axis=1)
    
    return pd.merge(revised_train, total, left_index=True, right_index=True, how='left')
        

def preprocessing(df, grp, text_replace_dic, script_version_dic):

    start, end = map(int,grp.split('-'))
    kol_lvl = (df.groupby(['session_id'])['level'].agg('nunique') < end - start + 1)
    list_session = kol_lvl[kol_lvl].index
    df = df[~df['session_id'].isin(list_session)]

    df = delt_time_def(df)
    
    train_ = feature_engineer(pl.from_pandas(df), grp, use_extra=False, feature_suffix='')
    
    # recap text count \w join
    train = text_cnt(df, train_)
    train['recap_reading'] = train['recap_reading'].fillna(0)
    
    # add year, month, day etc.
    train = time_feature(train)
    
    # new_page
    train = new_page(df, train, grp)
    
    #script_version
    train=fichi_1_2_3_4_python(df, train, text_replace_dic, script_version_dic)
    

    # # 상관관계 행렬 계산
    # correlation_matrix = train.corr().abs()
    # threshold = 0.9 # 임계값 설정

    # high_correlation_cols = np.where(correlation_matrix > threshold)
    # high_correlation_cols = [(correlation_matrix.columns[x], correlation_matrix.columns[y]) for x, y in zip(*high_correlation_cols) if x != y]

    # cols_to_drop = set()
    # for col1, col2 in high_correlation_cols:
    #     if col1 not in cols_to_drop and col2 not in cols_to_drop:
    #         cols_to_drop.add(col2)

    # # 중복 열 제거한 데이터프레임 생성
    # train = train.drop(cols_to_drop, axis=1)

    # add_text
    if grp =='0-4':
        train=add_text(df, train, 'text', group1_text_corr, 'corr_mean', 'corr_std', 'corr_max', 'corr_kur', 'corr_ske')
        train=add_text(df, train, 'text', group1_text_wron, 'wron_mean', 'wron_std', 'wron_max' ,'wron_kur', 'wron_ske')
        train = train.loc[:, ~train.columns.duplicated()] #중복열 삭제
        return train, df

    elif grp=='5-12':
        train= add_text(df, train, 'text', group2_text_corr, 'corr_mean', 'corr_std', 'corr_max','corr_kur', 'corr_ske' )
        train= add_text(df, train, 'text', group2_text_wron, 'wron_mean', 'wron_std', 'wron_max','wron_kur', 'wron_ske' )
        train = train.loc[:, ~train.columns.duplicated()] #중복열 삭제
        return train, df

    else:
        train=add_text(df, train, 'text', group3_text_wron, 'wron_mean', 'wron_std', 'wron_max','wron_kur', 'wron_ske' ) 
        train = train.loc[:, ~train.columns.duplicated()] #중복열 삭제
        return train, df
