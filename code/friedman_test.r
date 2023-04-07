#setwd('/dhcp/fmri_anna_graham/dhcp_camcan_timescales/results/')
setwd('C:/Users/Anna/Documents/Research/Projects/ONGOING/Project dHCP_Autocorrelation/dhcp_hcp_timescales/results')

### dhcp_group1 drop_scan
db <- read.csv('tau_estimation_dhcp_group1_7net_drop_scan_dhcp.csv', sep=',')
db_tau <- db[,4:403]
p95 <- quantile(db_tau,.95,na.rm=TRUE)
db_tau[db_tau>p95] <- NA
friedman.test(as.matrix(db_tau))



### dhcp_group2_drop_scan
db <- read.csv('tau_estimation_dhcp_group2_7net_drop_scan_dhcp.csv', sep=',')
db_tau <- db[,4:403]
p95 <- quantile(db_tau,.95,na.rm=TRUE)
db_tau[db_tau>p95] <- NA
friedman.test(as.matrix(db_tau))



### hcp
db <- read.csv('tau_estimation_hcp_7net_drop_scan_dhcp.csv', sep=',')
db_tau <- db[,3:402]
p95 <- quantile(db_tau,.95,na.rm=TRUE)
db_tau[db_tau>p95] <- NA
friedman.test(as.matrix(db_tau))

