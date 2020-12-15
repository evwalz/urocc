# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:01:18 2019

@author: walz
"""

from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

#  Weather variable paramID = "167" : 2m Temperature
#                           = "228228" : Precipitation
#                           = "165/166" : Wind speed (u and v component)      
             
def retrieve_tigge_data():
    paramID = "167"
    
    dates = ['2006-12-31/to/2007-01-30', '2007-01-31/to/2007-02-27','2007-02-28/to/2007-03-30','2007-03-31/to/2007-04-29','2007-04-30/to/2007-05-30','2007-05-31/to/2007-06-29',
             '2007-06-30/to/2007-07-30', '2007-07-31/to/2007-08-30','2007-08-31/to/2007-09-29','2007-09-30/to/2007-10-30','2007-10-31/to/2007-11-29','2007-11-30/to/2007-12-30',
             '2007-12-31/to/2008-01-30', '2008-01-31/to/2008-02-28','2008-02-29/to/2008-03-30','2008-03-31/to/2008-04-29','2008-04-30/to/2008-05-30','2008-05-31/to/2008-06-29',
             '2008-06-30/to/2008-07-30', '2008-07-31/to/2008-08-30','2008-08-31/to/2008-09-29','2008-09-30/to/2008-10-30','2008-10-31/to/2008-11-29','2008-11-30/to/2008-12-30',
             '2008-12-31/to/2009-01-30', '2009-01-31/to/2009-02-27','2009-02-28/to/2009-03-30','2009-03-31/to/2009-04-29','2009-04-30/to/2009-05-30','2009-05-31/to/2009-06-29',
             '2009-06-30/to/2009-07-30', '2009-07-31/to/2009-08-30','2009-08-31/to/2009-09-29','2009-09-30/to/2009-10-30','2009-10-31/to/2009-11-29','2009-11-30/to/2009-12-30',
             '2009-12-31/to/2010-01-30', '2010-01-31/to/2010-02-27','2010-02-28/to/2010-03-30','2010-03-31/to/2010-04-29','2010-04-30/to/2010-05-30','2010-05-31/to/2010-06-29',
             '2010-06-30/to/2010-07-30', '2010-07-31/to/2010-08-30','2010-08-31/to/2010-09-29','2010-09-30/to/2010-10-30','2010-10-31/to/2010-11-29','2010-11-30/to/2010-12-30',
             '2010-12-31/to/2011-01-30', '2011-01-31/to/2011-02-27','2011-02-28/to/2011-03-30','2011-03-31/to/2011-04-29','2011-04-30/to/2011-05-30','2011-05-31/to/2011-06-29',
             '2011-06-30/to/2011-07-30', '2011-07-31/to/2011-08-30','2011-08-31/to/2011-09-29','2011-09-30/to/2011-10-30','2011-10-31/to/2011-11-29','2011-11-30/to/2011-12-30',
             '2011-12-31/to/2012-01-30', '2012-01-31/to/2012-02-28','2012-02-29/to/2012-03-30','2012-03-31/to/2012-04-29','2012-04-30/to/2012-05-30','2012-05-31/to/2012-06-29',
             '2012-06-30/to/2012-07-30', '2012-07-31/to/2012-08-30','2012-08-31/to/2012-09-29','2012-09-30/to/2012-10-30','2012-10-31/to/2012-11-29','2012-11-30/to/2012-12-30',
             '2012-12-31/to/2013-01-30', '2013-01-31/to/2013-02-27','2013-02-28/to/2013-03-30','2013-03-31/to/2013-04-29','2013-04-30/to/2013-05-30','2013-05-31/to/2013-06-29',
             '2013-06-30/to/2013-07-30', '2013-07-31/to/2013-08-30','2013-08-31/to/2013-09-29','2013-09-30/to/2013-10-30','2013-10-31/to/2013-11-29','2013-11-30/to/2013-12-30',
             '2013-12-31/to/2014-01-30', '2014-01-31/to/2014-02-27','2014-02-28/to/2014-03-30','2014-03-31/to/2014-04-29','2014-04-30/to/2014-05-30','2014-05-31/to/2014-06-29',
             '2014-06-30/to/2014-07-30', '2014-07-31/to/2014-08-30','2014-08-31/to/2014-09-29','2014-09-30/to/2014-10-30','2014-10-31/to/2014-11-29','2014-11-30/to/2014-12-30',
             '2014-12-31/to/2015-01-30', '2015-01-31/to/2015-02-27','2015-02-28/to/2015-03-30','2015-03-31/to/2015-04-29','2015-04-30/to/2015-05-30','2015-05-31/to/2015-06-29',
             '2015-06-30/to/2015-07-30', '2015-07-31/to/2015-08-30','2015-08-31/to/2015-09-29','2015-09-30/to/2015-10-30','2015-10-31/to/2015-11-29','2015-11-30/to/2015-12-30',
             '2015-12-31/to/2016-01-30', '2016-01-31/to/2016-02-28','2016-02-29/to/2016-03-30','2016-03-31/to/2016-04-29','2016-04-30/to/2016-05-30','2016-05-31/to/2016-06-29',
             '2016-06-30/to/2016-07-30', '2016-07-31/to/2016-08-30','2016-08-31/to/2016-09-29','2016-09-30/to/2016-10-30','2016-10-31/to/2016-11-29','2016-11-30/to/2016-12-30',
             '2016-12-31/to/2017-01-30', '2017-01-31/to/2017-02-27','2017-02-28/to/2017-03-30','2017-03-31/to/2017-04-29','2017-04-30/to/2017-05-30','2017-05-31/to/2017-06-29',
             '2017-06-30/to/2017-07-30', '2017-07-31/to/2017-08-30','2017-08-31/to/2017-09-29','2017-09-30/to/2017-10-30','2017-10-31/to/2017-11-29','2017-11-30/to/2017-12-30',
             '2017-12-31/to/2018-01-30', '2018-01-31/to/2018-02-27','2018-02-28/to/2018-03-30','2018-03-31/to/2018-04-29','2018-04-30/to/2018-05-30','2018-05-31/to/2018-06-29',
             '2018-06-30/to/2018-07-30', '2018-07-31/to/2018-08-30','2018-08-31/to/2018-09-29','2018-09-30/to/2018-10-30','2018-10-31/to/2018-11-29','2018-11-30/to/2018-12-30'   ]
 
    for date in dates:
        datestemp = date[14:21]
        IDdatestemp = paramID + '_' + datestemp
        target = 'tiggeLead%s.nc' % (IDdatestemp)
        tigge_request(date, paramID, target)


def tigge_request(date, paramID, target):
        '''
       A TIGGE request for perturbed forecast, for 1 origins : ECMWF.
       Keep in mind that if you wish to download the same data, for more than one origins,
       it is more efficient to request all of them in one go.
       You can change the keywords below to adapt it to your needs,
       (ie to add more parameters, or steps, or even more origins etc),
       Presumably you need to check the availability of the requested origins.
    '''
        server.retrieve({
            "class": "ti",
            "dataset": "tigge",
            "date": date,
            "expver": "prod",
            "model": "glob",
            "levtype": "sfc",
            "grid": "0.25/0.25",
            "origin": "ecmf",
            "param": paramID,
            "step": "24",
            "time": "00:00:00",
            "type": "fc",
            "format": "netcdf",
            "target": target,
            })

          
if __name__ == '__main__':
    retrieve_tigge_data()
