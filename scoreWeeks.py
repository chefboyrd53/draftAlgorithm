import nfl_data_py as nfl
from scoring import yardageScoring, tdScoring, fgScoring, dstScoring
from localStorage import syncToFirebase, saveToFiles, loadFromFiles
from firebaseSetup import db
import sys

def scoreWeek(playByPlaydf, weeklydf, year, week):
    # Load existing data if available
    loadFromFiles()

    # yardage scoring and 2 point conversions
    playerRows = weeklydf[(weeklydf['week'] == week)]
    yardageScoring.scoreYardage(playerRows, week, year)
    print("did yardage scoring for", playerRows.shape[0], "players")

    # increment score/data with tds
    touchdownRows = playByPlaydf[(playByPlaydf['touchdown'] == 1) & (playByPlaydf['week'] == week)]
    tdScoring.score(touchdownRows, week, year)
    print("did touchdown scoring for", touchdownRows.shape[0], "rows")

    # field goals and extra points
    fgRows = playByPlaydf[((playByPlaydf['field_goal_result'] == 'made') | (playByPlaydf['extra_point_result'] == 'good')) & (playByPlaydf['week'] == week)]
    fgScoring.scoreFg(fgRows, week, year)
    print("did fg and extra point scoring for", fgRows.shape[0], "kicks")

    # defense and special teams scoring
    dstScoring.scoreDST(playByPlaydf, week, year)
    print("did defense and special teams scoring")
    
    # Save all data to files
    saveToFiles()

if __name__ == "__main__":
    year = int(sys.argv[1])

    playByPlaydf = nfl.import_pbp_data([year], downcast=False, cache=False, alt_path=None)
    weeklydf = nfl.import_weekly_data([year], downcast=False)

    for i in range(2, len(sys.argv)):
        scoreWeek(playByPlaydf, weeklydf, year, int(sys.argv[i]))
    
