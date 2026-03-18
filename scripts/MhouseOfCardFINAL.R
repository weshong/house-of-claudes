#import the files

#clear the environment
rm(list = ls())

require(stats)
require(e1071)
require(nFactors)
require(data.table)

# set data directory path
pathname_wes <- "~/Downloads/Kaggle2026/march-machine-learning-mania-2026"

setwd(pathname_wes)

# import .csv files from directory
files <- list.files(pattern = '*.csv')
for (i in 1:length(files)) {
  assign(files[i], read.csv(files[i]))
}

# global variables
SEASONS <- MSeasons.csv$Season



logLoss <- function(yhat, y) {
  # calculates log loss
  logLoss <- -1/length(yhat) * sum(y*log(yhat) + (1-y) * log(1-yhat))
  return(logLoss)
}


library(dplyr)
wp <- function(team_id, season = SEASONS, exclude = NULL, print_id = TRUE) {
  # calculates the winning percentage of a team
  # optional argument 'season' is a vector of seasons to calculate winning percentage
  # optional argument 'exclude' is a vector of opponent teams to exclude
  games <- MRegularSeasonCompactResults.csv[which(MRegularSeasonCompactResults.csv$Season %in% season),]
  wins <- length(games$WTeamID[which(games$WTeamID == team_id & !(games$LTeamID %in% exclude))])
  losses <- length(games$LTeamID[which(games$LTeamID == team_id & !(games$WTeamID %in% exclude))])
  wp <- wins / (wins + losses)
  if(print_id == TRUE) {
    print(team_id)
  }
  return(wp)
}


owp <- function(team_id, season = SEASONS, print_id = TRUE) {
  # calculates a descriptive statistic representing the distribution of winning percentages of a team's opponents
  # optional argument 'season' is a vector of seasons to calculate OWP
  games <- MRegularSeasonCompactResults.csv[which(MRegularSeasonCompactResults.csv$Season %in% season),]
  games <- games[which(games$WTeamID == team_id | games$LTeamID == team_id),]
  opponents <- c(games$WTeamID[which(games$WTeamID != team_id)],
                        games$LTeamID[which(games$LTeamID != team_id)])
  # median seems like a better statistic than average
  if(length(opponents) != 0) {
    owp <- median(sapply(opponents, wp, exclude = team_id, print_id = FALSE))  
  }
  else {
    owp <- NA
  }
  if(print_id == TRUE) {
    print(team_id)
  }
  return(owp)
}


pct <- function(stat, team_id, season = SEASONS, print_id = TRUE) {
  # calculates a descriptive statistic representing the team's shooting stats
  # stat is a string for the shooting statistic: fg, fg3, ft
  # optional argument 'season' is a vector of seasons to calculate over
  games <- MRegularSeasonDetailedResults.csv[which(MRegularSeasonDetailedResults.csv$Season %in% season),]
  Wgames <- games[which(games$WTeamID == team_id),]
  Lgames <- games[which(games$LTeamID == team_id),]
  stat.a <- gsub('^([A-Z]{2})', '\\1A\\2', stat)
  stat.m <- gsub('^([A-Z]{2})', '\\1M\\2', stat)
  pct <- (sum(Wgames[, paste('W', stat.m, sep = '')]) + sum(Lgames[, paste('L', stat.m, sep = '')])) /
    (sum(Wgames[, paste('W', stat.a, sep = '')]) + sum(Lgames[, paste('L', stat.a, sep = '')]))
  if(print_id == TRUE) {
    print(team_id)
  }
  return(pct)
}


average <- function(stat, team_id, season = SEASONS, print_id = TRUE) {
  # calculates a team's average non-shooting stat
  # stat is a string for the non-shooting statistic: or, dr, ast, to, stl, blk, pf
  # optional argument 'season' is a vector of seasons to calculate over
  games <- MRegularSeasonDetailedResults.csv[which(MRegularSeasonDetailedResults.csv$Season %in% season),]
  Wgames <- games[which(games$WTeamID == team_id),]
  Lgames <- games[which(games$LTeamID == team_id),]
  average <- (sum(Wgames[, paste('W', stat, sep = '')]) + sum(Lgames[, paste('L', stat, sep = '')])) /
    (length(Wgames[, paste('W', stat, sep = '')]) + length(Lgames[, paste('L', stat, sep = '')]))
  if(print_id == TRUE) {
    print(team_id)
  }
  return(average)
}



#JHopposing functions
opppct <- function(stat, team_id, season = SEASONS, print_id = TRUE) {
  # calculates a descriptive statistic representing the team's shooting stats
  # stat is a string for the shooting statistic: fg, fg3, ft
  # optional argument 'season' is a vector of seasons to calculate over
  games <- MRegularSeasonDetailedResults.csv[which(MRegularSeasonDetailedResults.csv$Season %in% season),]
  Wgames <- games[which(games$WTeamID == team_id),]
  Lgames <- games[which(games$LTeamID == team_id),]
  stat.a <- gsub('^([A-Z]{2})', '\\1A\\2', stat)
  stat.m <- gsub('^([A-Z]{2})', '\\1M\\2', stat)
  pct <- (sum(Wgames[, paste('L', stat.m, sep = '')]) + sum(Lgames[, paste('W', stat.m, sep = '')])) /
    (sum(Wgames[, paste('L', stat.a, sep = '')]) + sum(Lgames[, paste('W', stat.a, sep = '')]))
  if(print_id == TRUE) {
    print(team_id)
  }
  return(pct)
}


oppaverage <- function(stat, team_id, season = SEASONS, print_id = TRUE) {
  # calculates a team's average non-shooting stat
  # stat is a string for the non-shooting statistic: or, dr, ast, to, stl, blk, pf
  # optional argument 'season' is a vector of seasons to calculate over
  games <- MRegularSeasonDetailedResults.csv[which(MRegularSeasonDetailedResults.csv$Season %in% season),]
  Wgames <- games[which(games$WTeamID == team_id),]
  Lgames <- games[which(games$LTeamID == team_id),]
  average <- (sum(Wgames[, paste('L', stat, sep = '')]) + sum(Lgames[, paste('W', stat, sep = '')])) /
    (length(Wgames[, paste('L', stat, sep = '')]) + length(Lgames[, paste('W', stat, sep = '')]))
  if(print_id == TRUE) {
    print(team_id)
  }
  return(average)
}


#ENDJH





Mode <- function(x) {
  # calculates the mode of a vector x
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}


extract <- function(team_id, season, df, column) {
  # extracts value for a column name given the team_id and season
  row.no <- which(df$team_id == team_id & df$season == season)
  return(df[row.no, column])
}


#Define a function that extracts the seeds and divisions separately
getSeedDivision <- function(seedsInfo){
  #Seed & Division 
  #This function gets the seed and division of a team in a given season
  #Input class == "numeric" corresponding to the season of the tournament and the team unique ID
  #Returns class == "character" corresponding to the seed in that season and the division assigned in the tourney
  #seedsInfo <- tourneySeeds[1] #here for debugging
  
  seasonFromData <- seedsInfo[["Season"]]
  seedAndDivision <- seedsInfo[["Seed"]]
  teamFromData <- seedsInfo[["Team"]]
  
  seedTeam <- gsub(pattern = "[A-Z+a-z]", replacement = "", x = seedAndDivision)
  divisionTeam <- gsub(pattern = "[0-9]", replacement = "", x = seedAndDivision)
  #clean the extra letters
  divisionTeam <- gsub(pattern = "[a-z]", replacement = "", x = divisionTeam)  
  
  return(c(seasonFromData, teamFromData, seedTeam, divisionTeam))
}




# flag for running script
if(TRUE) {
  
  # Script for calculating the WP and OWP of each team, each season
  team_id <- MTeams.csv$TeamID
  KeyStats <- data.frame(matrix(ncol = 4, nrow = length(team_id)*length(SEASONS)))
  colnames(KeyStats) <- c('team_id', 'season', 'wp', 'owp')
  KeyStats$team_id <- rep(team_id, each = length(SEASONS))
  KeyStats$season <- rep(SEASONS, times = length(team_id))
  KeyStats$wp <- mapply(wp, team_id = KeyStats$team_id, season = KeyStats$season)
  KeyStats$owp <- mapply(owp, team_id = KeyStats$team_id, season = KeyStats$season)
  
    
    
}


# flag for running script
if(TRUE) {
  
  # Script for calculating season efficiency stats
  print('fg')
  KeyStats$fg <- mapply(pct, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FG')
  print('fg3')
  KeyStats$fg3 <- mapply(pct, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FG3')
  print('ft')
  KeyStats$ft <- mapply(pct, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FT')
  print('or')
  KeyStats$oreb <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'OR')
  print('dr')
  KeyStats$dreb <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'DR')
  print('ast')
  KeyStats$ast <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Ast')
  print('to')
  KeyStats$to <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'TO')
  print('stl')
  KeyStats$stl <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Stl')
  print('blk')
  KeyStats$blk <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Blk')
  print('pf')
  KeyStats$pf <- mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'PF')
  
  #JHCalc the opposing teams out.
  print('fg')
  KeyStats$oppfg <- mapply(opppct, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FG')
  print('fg3')
  KeyStats$oppfg3 <- mapply(opppct, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FG3')
  print('ft')
  KeyStats$oppft <- mapply(opppct, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FT')
  print('or')
  KeyStats$opporeb <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'OR')
  print('dr')
  KeyStats$oppdreb <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'DR')
  print('ast')
  KeyStats$oppast <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Ast')
  print('to')
  KeyStats$oppto <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'TO')
  print('stl')
  KeyStats$oppstl <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Stl')
  print('blk')
  KeyStats$oppblk <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Blk')
  print('pf')
  KeyStats$opppf <- mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'PF')
  

  
  #JHcalculate possessions/game and then efficiency stats
  KeyStats$fga <-  mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FGA')
  KeyStats$oppfga <-  mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FGA')
  KeyStats$fta <-  mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FTA')
  KeyStats$oppfta <-  mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FTA')
  
  
  KeyStats$pt <-  mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Score')
  KeyStats$opppt <-  mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'Score')
  
  KeyStats$pos <- KeyStats$fga - KeyStats$oreb + KeyStats$to + 0.475 * KeyStats$fta
  KeyStats$opppos <- KeyStats$oppfga - KeyStats$opporeb + KeyStats$oppto + 0.475 * KeyStats$oppfta
  
  KeyStats$oeff <- KeyStats$pt/KeyStats$pos
  KeyStats$deff <- KeyStats$opppt/KeyStats$opppos
  
  KeyStats$fgm <-  mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FGM')
  KeyStats$oppfgm <-  mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FGM')
  KeyStats$fgm3 <-  mapply(average, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FGM3')
  KeyStats$oppfgm3 <-  mapply(oppaverage, team_id = KeyStats$team_id, season = KeyStats$season, stat = 'FGM3')
  
  #need to also define the "four factors" (dean oliver)
  #effective FG = (FGM + 0.5 * 3s made)/FGA
  KeyStats$effFG <- (KeyStats$fgm + (0.5 * KeyStats$fgm3))/KeyStats$fga
  KeyStats$oppeffFG <-(KeyStats$oppfgm + (0.5 * KeyStats$oppfgm3))/KeyStats$oppfga
  
  #turnover %
  KeyStats$topct <- KeyStats$to/KeyStats$pos
  KeyStats$opptopct <- KeyStats$oppto/KeyStats$opppos
  
  #offensive rebound %
  KeyStats$orebpct <- KeyStats$oreb/(KeyStats$oreb + KeyStats$oppdreb)
  KeyStats$opporebpct <- KeyStats$opporeb/(KeyStats$opporeb + KeyStats$dreb)
  
  #free throw rate
  KeyStats$ftr <- KeyStats$fta/KeyStats$fga
  KeyStats$oppftr <- KeyStats$oppfta/KeyStats$oppfga

  
  
}



# flag for running script
if(TRUE) {
 
  # Script for preparing training/testing dataframe
  RegularSeason <- data.frame(matrix(ncol = 5, 
                               nrow = length(MRegularSeasonDetailedResults.csv[, 1])))
  colnames(RegularSeason) <- c('teamA',
                         'teamB',
                         'season',
                         'win',
                         'spread')
  
  RegularSeason$teamA <- c(MRegularSeasonDetailedResults.csv$WTeamID[
      which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
    MRegularSeasonDetailedResults.csv$LTeamID[
      which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  
  RegularSeason$season <- c(MRegularSeasonDetailedResults.csv$Season[
      which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
    MRegularSeasonDetailedResults.csv$Season[
      which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  
  RegularSeason$teamB <- c(MRegularSeasonDetailedResults.csv$LTeamID[
    which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
    MRegularSeasonDetailedResults.csv$WTeamID[
      which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  
  RegularSeason$win <- factor(c(rep('W', length(which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID))),
                   rep('L', length(which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)))),
                   levels = c('L','W'))
  RegularSeason$spread <- c(MRegularSeasonDetailedResults.csv$WScore[
      which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
    MRegularSeasonDetailedResults.csv$LScore[
      which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)]) -
  c(MRegularSeasonDetailedResults.csv$LScore[
      which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
    MRegularSeasonDetailedResults.csv$WScore[
      which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  
  # 
  # RegularSeason$teamApt <- c(MRegularSeasonDetailedResults.csv$Wscore[
  #   which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
  #   MRegularSeasonDetailedResults.csv$Lscore[
  #     which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  # RegularSeason$teamBpt <- c(MRegularSeasonDetailedResults.csv$Lscore[
  #   which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
  #   MRegularSeasonDetailedResults.csv$Wscore[
  #     which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  # # 
  # #new - need to add the dates in to the structure too
  # RegularSeason$Daynum <- c(MRegularSeasonDetailedResults.csv$Daynum[
  #   which(MRegularSeasonDetailedResults.csv$WTeamID < MRegularSeasonDetailedResults.csv$LTeamID)],
  #   MRegularSeasonDetailedResults.csv$Daynum[
  #     which(MRegularSeasonDetailedResults.csv$LTeamID < MRegularSeasonDetailedResults.csv$WTeamID)])
  
  
}

# flag for running script
if(TRUE) {
  
  # Script for preparing training/testing dataframe; similarly creating the tourney dataframe
  Tourney <- data.frame(matrix(ncol = 5, 
                                     nrow = length(MNCAATourneyDetailedResults.csv[, 1])))
  colnames(Tourney) <- c('teamA',
                               'teamB',
                               'season',
                               'win',
                               'spread')
  
  Tourney$teamA <- c(MNCAATourneyDetailedResults.csv$WTeamID[
    which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
    MNCAATourneyDetailedResults.csv$LTeamID[
      which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  
  Tourney$season <- c(MNCAATourneyDetailedResults.csv$Season[
    which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
    MNCAATourneyDetailedResults.csv$Season[
      which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  
  Tourney$teamB <- c(MNCAATourneyDetailedResults.csv$LTeamID[
    which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
    MNCAATourneyDetailedResults.csv$WTeamID[
      which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  
  Tourney$win <- factor(c(rep('W', length(which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID))),
                                rep('L', length(which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)))),
                              levels = c('L','W'))
  Tourney$spread <- c(MNCAATourneyDetailedResults.csv$WScore[
    which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
    MNCAATourneyDetailedResults.csv$LScore[
      which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)]) -
    c(MNCAATourneyDetailedResults.csv$LScore[
      which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
      MNCAATourneyDetailedResults.csv$WScore[
        which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  
  
  # Tourney$teamApt <- c(MNCAATourneyDetailedResults.csv$Wscore[
  #   which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
  #   MNCAATourneyDetailedResults.csv$Lscore[
  #     which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  # Tourney$teamBpt <- c(MNCAATourneyDetailedResults.csv$Lscore[
  #   which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
  #   MNCAATourneyDetailedResults.csv$Wscore[
  #     which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  # 
  
  #new - need to add the dates in to the structure too
  # Tourney$Daynum <- c(MNCAATourneyDetailedResults.csv$Daynum[
  #   which(MNCAATourneyDetailedResults.csv$WTeamID < MNCAATourneyDetailedResults.csv$LTeamID)],
  #   MNCAATourneyDetailedResults.csv$Daynum[
  #     which(MNCAATourneyDetailedResults.csv$LTeamID < MNCAATourneyDetailedResults.csv$WTeamID)])
  # 
}

# #calculate the teams elo ranking for KeyStats
# library(plyr)
# RegularSeason <- rename(RegularSeason, c("season"="Season"))
# RegularSeason <- merge(Seasons.csv[,c("Season","Dayzero")], RegularSeason, by="Season")
# 
# #then the date needs to be converted
# RegularSeason$Dayzero <- strptime(RegularSeason$Dayzero, "%m/%d/%Y") 
# library(lubridate)
# RegularSeason$Day <- RegularSeason$Dayzero + days(RegularSeason$Daynum)
# 
# #cleanup
# RegularSeason$Dayzero <- NULL
# RegularSeason$Daynum <- NULL
# 
# #calculate the teams elo ranking for KeyStats
# Tourney <- rename(Tourney, c("season"="Season"))
# Tourney <- merge(Seasons.csv[,c("Season","Dayzero")], Tourney, by="Season")
# 
# #then the date needs to be converted
# Tourney$Dayzero <- strptime(Tourney$Dayzero, "%m/%d/%Y") 
# library(lubridate)
# Tourney$Day <- Tourney$Dayzero + days(Tourney$Daynum)
# 
# #cleanup
# Tourney$Dayzero <- NULL
# Tourney$Daynum <- NULL
# 
# #first make a separate matrix combining the two and merge
# outcomes <- rbind(RegularSeason[c("teamA", "teamB", "teamApt", "teamBpt", "win", "Day")], 
#                   Tourney[c("teamA", "teamB", "teamApt", "teamBpt", "win", "Day")])
# 
# 
# #then make a winners column
# outcomes$winner <- ifelse(outcomes$win == "W", outcomes$teamA, outcomes$teamB)
# outcomes$loser <- ifelse(outcomes$win == "W", outcomes$teamB, outcomes$teamA)
# 
# #apparently this needs to be sorted
# outcomes <- outcomes[order(outcomes$Day),] 
# 


# library(EloRating)
# eloresult <- elo.seq(winner = outcomes$winner, loser = outcomes$loser, Date = outcomes$Day)
# 
# #now get this into regular season and tourney; for some reason it has to be in a loop (no idea why)
# for (i in 1:nrow(RegularSeason)){
#   RegularSeason$teamAElo[i] <- unname(extract.elo(eloresult, RegularSeason$Day[i], IDs = as.character(RegularSeason$teamA[i])))
# }
# for (i in 1:nrow(RegularSeason)){
#   RegularSeason$teamBElo[i] <- unname(extract.elo(eloresult, RegularSeason$Day[i], IDs = as.character(RegularSeason$teamB[i])))
# }
# for (i in 1:nrow(Tourney)){
#   Tourney$teamAElo[i] <- unname(extract.elo(eloresult, Tourney$Day[i], IDs = as.character(Tourney$teamA[i])))
# }
# for (i in 1:nrow(Tourney)){
#   Tourney$teamBElo[i] <- unname(extract.elo(eloresult, Tourney$Day[i], IDs = as.character(Tourney$teamB[i])))
# }
# 
# #check by sorting by date
# temp <- RegularSeason[order(RegularSeason$Day),] 
# 
# #check visualization of elo
# #eloplot(eloresult, ids = c("1343", "1196"), from = "2017-01-25", to = "2017-02-14")
# 
# #clean up by removing the date
# RegularSeason$Day <- NULL
# Tourney$Day <- NULL
#######

#incorporate the keystats into each game for RF; x is teamA and y is teamB
#1st step is to flip the teams though too.
#make a new dataframe first
library(plyr)
invert <- RegularSeason
invert <- rename(invert, c("teamA" = "teamB", "teamB" = "teamA"))
invert <- invert[c(colnames(RegularSeason))]
RegularSeason$tourney <- 0
invert$tourney <- 0
invert$win[invert$win == "W"] <- NA
invert$win[invert$win == "L"] <- "W"
invert$win[is.na(invert$win)] <- "L"
invert$spread <- -1*invert$spread


inverttourney <- Tourney
inverttourney <- rename(inverttourney, c("teamA" = "teamB", "teamB" = "teamA"))
inverttourney <- inverttourney[colnames(Tourney)]
Tourney$tourney <- 1
inverttourney$tourney <- 1
inverttourney$win[inverttourney$win == "W"] <- NA
inverttourney$win[inverttourney$win == "L"] <- "W"
inverttourney$win[is.na(inverttourney$win)] <- "L"
inverttourney$spread <- -1*inverttourney$spread

#combine all the data together
RFdata <- rbind(RegularSeason, invert, Tourney, inverttourney)

#incorporate seeds
RFdata<-merge.data.frame(RFdata, MNCAATourneySeeds.csv, by.x = c("teamA", "season"), 
                            by.y = c("TeamID", "Season"), all.x = TRUE)
RFdata<-merge.data.frame(RFdata, MNCAATourneySeeds.csv, by.x = c("teamB", "season"), 
                            by.y = c("TeamID", "Season"), all.x = TRUE)
RFdata$Seed.x <- as.numeric(substr(RFdata$Seed.x, 2,3))
RFdata$Seed.y <- as.numeric(substr(RFdata$Seed.y, 2,3))
RFdata$Seed.x <- ifelse(is.na(RFdata$Seed.x), 99, RFdata$Seed.x)
RFdata$Seed.y <- ifelse(is.na(RFdata$Seed.y), 99, RFdata$Seed.y)

filtKeyStats <- na.omit(KeyStats)
RFdata<-merge.data.frame(RFdata, filtKeyStats, by.x = c("teamA", "season"), by.y = c("team_id", "season"))
RFdata<-merge.data.frame(RFdata, filtKeyStats, by.x = c("teamB", "season"), by.y = c("team_id", "season"))
rm(filtKeyStats)

#final cleanup; this leaves only the win and spread of the game as far as game results
RFdata$teamA <- NULL
RFdata$teamB <- NULL
# RFdata$teamApt <- NULL
# RFdata$teamBpt <- NULL
RFdata$season <- NULL

#############
#random forest for win (built off of RFdata); this is how we did it in 2016; just need to drop spread
RFdataWIN <- RFdata
RFdataWIN$spread <- NULL

# all games
library(randomForest)

set.seed(313)
testRFWIN <- randomForest(win ~., data = RFdataWIN, ntree = 1000, do.trace = TRUE) #my guess is this will take about an hour

varImpPlot(testRFWIN)

# new for 2023 - add a logistic regression wrapper for calibration
temp <- predict(testRFWIN, RFdataWIN, type = "prob")
calibtable <- data.frame(matrix(ncol = 2, nrow = nrow(RFdataWIN)))
names(calibtable)[1] <- "win"
names(calibtable)[2] <- "RF"
calibtable$win <- ifelse(RFdataWIN$win == "W", 1, 0)
calibtable$RF <- temp[,2]

#check calibration plot of the RF on the training data
library(ggplot2)
ggplot(calibtable, aes(RF, win)) +
  geom_point(shape = 21, size = 2) +
  geom_abline(slope = 1, intercept = 0) +
  geom_smooth(method = stats::loess, se = FALSE) +
  scale_x_continuous(breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  xlab("Estimated Prob.") +
  ylab("Data w/ Empirical Prob.") +
  ggtitle("Logistic Regression Calibration Plot")



calibRFWIN <- glm(RFdataWIN$win ~.,family=binomial(link='logit'),data= calibtable)

######
#xgboost
library(Matrix)
library(xgboost)
library(caret)
xgtarget <-  as.numeric(RFdataWIN$win)-1
temp <- RFdataWIN
temp$win <- NULL
xgtrain <- model.matrix(~.+0,data = temp)


xgparams <- list(booster = "gbtree", objective = "multi:softprob", eta=0.01, gamma=0,
                 min_child_weight=1, eval_metric = "mlogloss")


#hyperparameter search
#make sure to save before running this since this can crash/take a while
searchGridSubCol <- expand.grid(subsample = c(0.5, 0.75, 1),
                                colsample_bytree = c(0.4, 0.6, 0.8, 1)
                                #max.depth = c(4, 6, 8, 10)
)

ntrees <- 1000
rm(aucHyperparameters, xgboostModelCV)

set.seed(313)
aucHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  #currentmax.depth <- parameterList[["max.depth"]] #new addition
  
  xgboostModelCV <- xgb.cv(data =  xgtrain, nthread = 10,
                           label = xgtarget,
                           nrounds = ntrees,
                           nfold = 5,
                           prediction = TRUE,
                           showsd = TRUE,
                           statified = TRUE,
                           print_every_n = 1,
                           early_stop_round = 10,
                           
                           gamma = 0,
                           min_child_weight = 1,
                           booster = "gbtree",
                           
                           metrics = "logloss",
                           verbose = TRUE,
                           "eval_metric" = "logloss",
                           "objective" = "binary:logistic",
                           "max.depth" = 6,
                           "eta" = 0.01,
                           "subsample" = currentSubsampleRate,
                           "colsample_bytree" = currentColsampleRate)
  
  #Save auc of the last iteration
  logloss <- tail(xgboostModelCV$evaluation_log$test_logloss_mean, 1)
  gc()
  
  return(c(logloss, currentSubsampleRate, currentColsampleRate))
  
})

library(ggplot2)
# Basic scatter plot
ggplot(as.data.frame(t(aucHyperparameters)), aes(x=paste(as.character(V2), as.character(V3)), y=V1, "")) + geom_point()
View(as.data.frame(t(aucHyperparameters)))



#1 and 0.6 have the lowest log loss

tunedsub <- 1
tunedcolsamp <- 0.6
max.depth <- 6

set.seed(313)
# fit the model with the tuned parameters
xgb_tuned <- xgboost(data = xgtrain, label = xgtarget,
                     params = xgparams,
                     nrounds = 1000, # max number of trees to build
                     verbose = TRUE,
                     print_every_n = 1,
                     early_stop_round = 10, # stop if no improvement within 10 trees
                     num_class = 2,
                     subsample=tunedsub, colsample_bytree=tunedcolsamp, max.depth = max.depth #tuned
)

# cross-validate xgboost to assess more accurate error
set.seed(220)
xgb_cv_tuned <- xgb.cv(params = xgparams, nthreads = 10,
                       data = xgtrain, label = xgtarget,
                       nrounds = 1000,
                       num_class = 2,
                       nfold = 5, # number of folds in K-fold
                       prediction = TRUE, # return the prediction using the final model
                       showsd = TRUE,  # standard deviation of loss across folds
                       stratified = TRUE, # sample is unbalanced; use stratified sampling
                       verbose = TRUE,
                       print_every_n = 1,
                       early_stop_round = 10,
                       subsample=tunedsub, colsample_bytree=tunedcolsamp, max.depth = max.depth #these are parameters that usually need tuning
)


#view variable importance plot
mat <- xgb.importance (feature_names = colnames(xgtrain), model = xgb_tuned)
xgb.plot.importance (importance_matrix = mat[1:20])


#########
#create output
output <- SampleSubmissionStage2.csv
output$season <- as.numeric(substr(output$ID,1,4))
output$team1 <- as.numeric(substr(output$ID,6,9))
output$team2 <- as.numeric(substr(output$ID,11,14))

output<-merge.data.frame(output, MNCAATourneySeeds.csv, by.x = c("team1", "season"), 
                            by.y = c("TeamID", "Season"), all.x = TRUE)
output<-merge.data.frame(output, MNCAATourneySeeds.csv, by.x = c("team2", "season"), 
                            by.y = c("TeamID", "Season"), all.x = TRUE)
output$Seed.x <- as.numeric(substr(output$Seed.x, 2,3))
output$Seed.y <- as.numeric(substr(output$Seed.y, 2,3))
output$Seed.x <- ifelse(is.na(output$Seed.x), 99, output$Seed.x)
output$Seed.y <- ifelse(is.na(output$Seed.y), 99, output$Seed.y)

filtKeyStats <- na.omit(KeyStats)
output<-merge.data.frame(output, filtKeyStats, by.x = c("team1", "season"), by.y = c("team_id", "season"))
output<-merge.data.frame(output, filtKeyStats, by.x = c("team2", "season"), by.y = c("team_id", "season"))
rm(filtKeyStats)
output$tourney <- 1


RFoutput <- output
XGBoutput <- output
oldRFoutput <- output




#write projections
RFoutput$ourproj <- predict(testRFWIN, RFoutput)
temp <- predict(testRFWIN, RFoutput, type = "prob")
RFoutput$winprob <- temp[,2]
RFoutput$RF <- RFoutput$winprob

#also write the calibrated version
temp <- predict(calibRFWIN, RFoutput, type = "response")
RFoutput$calibprob <- temp

#bring in team names too
RFoutput<-merge.data.frame(RFoutput, MTeams.csv, by.x = c("team1"), by.y = c("TeamID"))
RFoutput<-merge.data.frame(RFoutput, MTeams.csv, by.x = c("team2"), by.y = c("TeamID"))

#now make two options from RFoutput based on opposite sides of the bracket
#first ID the opposite sides of the bracket based on TourneySeeds. 
#W/E VS S/MW
#be careful with this bc the naming changes are in alphabetical order
#so East = W, Midwest = X, South = Y, West = Z
#this means west/east is W and Z

region <- MNCAATourneySeeds.csv %>% filter(Season == 2022)
region$region <- substr(region$Seed,1,1)

RFoutput <- left_join(RFoutput, region, by = c("team1" = "TeamID"))
RFoutput <- left_join(RFoutput, region, by = c("team2" = "TeamID"))

RFoutputsub1 <- RFoutput
RFoutputsub2 <- RFoutput

#W will always play X in the semis and Y will always play Z, so need the combinations
#where those matchups would be finals (W or X vs Y or Z)
RFoutputsub1$winprob[(RFoutput$region.x == 'W' & RFoutput$region.y == 'Y') |
                     (RFoutput$region.x == 'Y' & RFoutput$region.y == 'W') |
                     (RFoutput$region.x == 'W' & RFoutput$region.y == 'Z') |
                     (RFoutput$region.x == 'Z' & RFoutput$region.y == 'W') |
                     (RFoutput$region.x == 'X' & RFoutput$region.y == 'Y') |
                     (RFoutput$region.x == 'Y' & RFoutput$region.y == 'X') |
                     (RFoutput$region.x == 'X' & RFoutput$region.y == 'Z') |
                     (RFoutput$region.x == 'Z' & RFoutput$region.y == 'X')] <- 1

#then make all 0
RFoutputsub2$winprob[(RFoutput$region.x == 'W' & RFoutput$region.y == 'Y') |
                     (RFoutput$region.x == 'Y' & RFoutput$region.y == 'W') |
                     (RFoutput$region.x == 'W' & RFoutput$region.y == 'Z') |
                     (RFoutput$region.x == 'Z' & RFoutput$region.y == 'W') |
                     (RFoutput$region.x == 'X' & RFoutput$region.y == 'Y') |
                     (RFoutput$region.x == 'Y' & RFoutput$region.y == 'X') |
                     (RFoutput$region.x == 'X' & RFoutput$region.y == 'Z') |
                     (RFoutput$region.x == 'Z' & RFoutput$region.y == 'X')] <- 0

#output just appropriate things to table
temp <- data.frame(RFoutput$ID, RFoutput$winprob)
temp <- rename(temp, c("RFoutput.ID" = "ID", "RFoutput.winprob" = "Pred"))
write.csv(temp, "menRFourSubmission.csv", row.names = FALSE)
write.csv(RFoutput, "menRFoutputforwes.csv", row.names = FALSE)

temp <- data.frame(RFoutput$ID, RFoutput$calibprob)
temp <- rename(temp, c("RFoutput.ID" = "ID", "RFoutput.calibprob" = "Pred"))
write.csv(temp, "menCalRFourSubmission.csv", row.names = FALSE)


temp <- data.frame(RFoutputsub1$ID, RFoutputsub1$winprob)
temp <- rename(temp, c("RFoutputsub1.ID" = "ID", "RFoutputsub1.winprob" = "Pred"))
write.csv(temp, "menRFourSubmission1.csv", row.names = FALSE)


temp <- data.frame(RFoutputsub2$ID, RFoutputsub2$winprob)
temp <- rename(temp, c("RFoutputsub2.ID" = "ID", "RFoutputsub2.winprob" = "Pred"))
write.csv(temp, "menRFourSubmission2.csv", row.names = FALSE)



#######XGB
#write projections
#clean up the output matrix
#final cleanup; this leaves only the win and spread of the game as far as game results
XGBoutput$team1 <- NULL
XGBoutput$team2 <- NULL
XGBoutput$season <- NULL
XGBoutput$ID <- NULL

#then resort
col.order <- colnames(xgtrain)

library(dplyr)
xgoutput <- model.matrix(~.+0,data = select(XGBoutput, - Pred))
xgoutput <- xgoutput[,col.order]

xgourproj <- predict(xgb_tuned, xgoutput)

library(dplyr)
test_prediction <- 
  matrix(xgourproj, nrow = 2, ncol=length(xgourproj)/2) %>%
  t() %>%
  data.frame()

XGBoutput$Pred <- test_prediction$X2

#bring in team names too
XGBoutput$team2 <- output$team2
XGBoutput$season <- output$season
XGBoutput$team1 <- output$team1
XGBoutput$ID <- output$ID

#bring in team names too
XGBoutput<-merge.data.frame(XGBoutput, MTeams.csv, by.x = c("team1"), by.y = c("TeamID"))
XGBoutput<-merge.data.frame(XGBoutput, MTeams.csv, by.x = c("team2"), by.y = c("TeamID"))

#output just appropriate things to table
temp <- data.frame(XGBoutput$ID, XGBoutput$Pred)
write.csv(temp, "XGBourSubmission.csv", row.names = FALSE)
write.csv(XGBoutput, "XGBoutputforwes.csv", row.names = FALSE)

####
#other junk
#testing against higher seed
validationWin$higherseed <- ifelse(validationWin$Seed.x < validationWin$Seed.y, "W", "L")
validationWin$ourproj <- predict(testRFWin, validationWin)
temp <- predict(testRFWin, validationWin, type = "vote")
validationWin$winprob <- temp[,2]

validationWin$ourcorrect <- ifelse(validationWin$win == validationWin$ourproj, 1, 0)
validationWin$seedcorrect <- ifelse(validationWin$win == validationWin$higherseed, 1, 0)

sum(validationWin$seedcorrect)/length(validationWin$seedcorrect)
sum(validationWin$ourcorrect)/length(validationWin$ourcorrect)



validationWin$correct <- ifelse(validationWin$win == validationWin$proj, 1, 0)
temp <- validationWin$winprob[which(validationWin$correct == 0)]
