# NTSB航空事故数据库
## 数据概况
- 数据来源：美国国家运输安全委员会 (NTSB)
- 下载地址：https://data.ntsb.gov/avdata
- 文件格式：Microsoft Access (.mdb)
- 覆盖范围：2008年至今的航空事故记录

## 关键字段（参考）
- 事故日期与时间 (Event Date)
- 事故地点 (Location - City, State)
- 航空器型号 (Aircraft Make/Model)
- 运营人 (Operator)
- 天气条件 (Weather Conditions)
- 损伤程度 (Damage)
- 可能原因 (Probable Cause)
- 伤亡情况 (Injuries/Fatalities)
- 飞行阶段 (Flight Phase)

## 使用说明
本数据为MS Access格式(.mdb)，需安装pyodbc后转换: pip install pyodbc
或访问 https://www.ntsb.gov/Pages/AviationQueryV2.aspx 手动导出CSV