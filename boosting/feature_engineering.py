import pandas as pd
import polars as pl

# First model (Catboost)
CATS = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
NUMS = ['page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
        'hover_duration', 'elapsed_time_diff']
DIALOGS = ['that', 'this', 'it', 'you','find','found','Found','notebook','Wells','wells','help','need', 'Oh','Ooh','Jo', 'flag', 'can','and','is','the','to']

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
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).sum().alias(f'word_sum_{c}') for c in
          DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).median().alias(f'word_median_{c}') for c
          in DIALOGS],

        *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in CATS],

        *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in NUMS],
        *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in NUMS],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in NUMS],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in NUMS],
        *[pl.col(c).median().alias(f"{c}_median_{feature_suffix}") for c in NUMS],

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
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in fqid_lists],

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
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in text_lists],

        *[pl.col("room_fqid").filter(pl.col("room_fqid") == c).count().alias(f"{c}_room_fqid_counts{feature_suffix}")
          for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).median().alias(f"{c}_ET_median_{feature_suffix}")
          for
          c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in room_lists],

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
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in
          name_feature],

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
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in
          LEVELS],

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
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for
          c in
          level_groups],

    ]
    #df = x.groupby(['session_id']).agg(aggs).sort_values("session_id")

    df = x.groupby(['session_id'], maintain_order=True).agg(aggs).sort("session_id")

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
    train["year"] = train["session_id"].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
    train["month"] = train["session_id"].apply(lambda x: int(str(x)[2:4])+1).astype(np.uint8)
    train["day"] = train["session_id"].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
    train["hour"] = train["session_id"].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
    train["minute"] = train["session_id"].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
    train["second"] = train["session_id"].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)
    return train


def new_page(X, grp): 
    '''
    X= revised_train dataset
    '''
    # 이상치 session_id 추출
    if grp == '5-12':
        session_2=X[X.page==0].session_id.unique().tolist()
        X.loc[X['session_id'].isin(session_2), 'new_page'] = 0
        X.loc[~X['session_id'].isin(session_2), 'new_page'] = 1
    
    if grp == '13-22':
        session_3=X[(X.page==0)|(X.page==1)|(X.page==2)].session_id.unique().tolist()
        X.loc[X['session_id'].isin(session_3), 'new_page'] = 0
        X.loc[~X['session_id'].isin(session_3), 'new_page'] = 1
    
    return X.groupby(['session_id']).first().reset_index()

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
    
    return pd.merge(revised_train,text_feature, on='session_id', how='left').set_index('session_id')