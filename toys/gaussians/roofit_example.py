from ROOT import RooRealVar, RooGaussian, RooChebychev, RooAddPdf, RooArgList, RooArgSet, RooFit


x = RooRealVar("x","x",-1,1)

# Use RooGaussian in the generation
mean = RooRealVar("mean","mean of gaussian",0,-1,1)
sigma = RooRealVar("sigma","sigma of gaussian",0.1,-1,1)
sig = RooGaussian("gauss","gaussian PDF",x,mean,sigma) ;

# Background
a0 = RooRealVar("a0","a0",0.5,0.,1.)
a1 = RooRealVar("a1","a1",-0.2,0.,1.)
bkg = RooChebychev("bkg","Background",x,RooArgList(a0,a1))

bkgfrac = RooRealVar("bkgfrac","fraction of background",0.5,0.,1.)
model = RooAddPdf("model","g+a",RooArgList(bkg,sig), RooArgList(bkgfrac) )

data = model.generate(RooArgSet(x), 10000)

model.fitTo(data)