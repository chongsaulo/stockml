from services.stock_analytic_service import StockAnalyticService

stock_analytic_service = StockAnalyticService(
    stock_code='^HSI', technical_only=True)
# stock_analytic_service.exportStockData()
stock_analytic_service.buildTrain()
stock_analytic_service.shapeTrainData(0.25)
stock_analytic_service.buildModel()
stock_analytic_service.train()
stock_analytic_service.evaluate()
