from datetime import date
from utils.configloader import ConfigUtil

AI_BUDGET_TABLE = ConfigUtil.getInstance().configJSON["db.budgetTable"]
AI_TRAIN_LOG_TABLE = ConfigUtil.getInstance().configJSON["db.aiTrainLogTableName"]

INSERT_AI_BUDGET_SQL = "INSERT INTO %s (CustomerID, Week, AllocatedAmount, CategoryTypeCode, Spending) values(?,?,?,?,?)" %AI_BUDGET_TABLE

INSERT_AI_TRAIN_LOG_SQL = "INSERT INTO %s (ModelName, TrainingDate, ModelScore) values(?,?,?)" %AI_TRAIN_LOG_TABLE