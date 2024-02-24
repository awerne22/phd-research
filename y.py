#from jupyter_dash import JupyterDash

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import sympy as sm
import numpy as np
import pandas as pd
import scipy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import dash_bootstrap_components as dbc

def f(xi,y,u):
    return u

def calc(xi1,g,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    def rk(xi,y,u,h,g,xi1,coef):
        k1=h*f(xi,y,u)
        l1=h*g(xi,y,u,xi1,mu_ser,coef)
        k2=h*f(xi+h/2,y+k1/2,u+l1/2)
        l2=h*g(xi+h/2,y+k1/2,u+l1/2,xi1,mu_ser,coef)
        k3=h*f(xi+h/2,y+k2/2,u+l2/2)
        l3=h*g(xi+h/2,y+k2/2,u+l2/2,xi1,mu_ser,coef)
        k4=h*f(xi+h,y+k3,u+l3)
        l4=h*g(xi+h,y+k3,u+l3,xi1,mu_ser,coef)
        return y0+1/6*(k1+2*k2+2*k3+k4),u0+1/6*(l1+2*l2+2*l3+l4)
    h=1e-3
    y0=1
    u0=0
    xi0=0
    y=[y0]
    du=[u0]
    xi=[xi0]
    coef=a0,a1,a2,a3,a4,b0,b1,b2,b3,b4
    try:
        while y[-1]>0:
            y0,u0=rk(xi0,y0,u0,h,g,xi1,coef)
            xi0+=h
            y.append(y0)
            xi.append(xi0)
            du.append(u0)
        return np.array(xi),np.array(y)
    except ValueError:
        return print(y)
X_=0.708
Z_=0.020
x_=sm.symbols("x")
def X(x,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    #a0=4.50096e-05
    #a1=-0.000669583
    #a2=0.0358607 
    #a3=-0.53596 
    #a4=4.97482
    #b0=0.000448987
    #b1=-0.00399509
    #b2=0.0745353
    #b3=-0.827868
    #b4=7.08559

    try:    
        return ((a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x)/(b0+b1*x+b2*x*x+b3*x*x*x+b4*x*x*x*x))
    except TypeError:
        return 0
        
data={}
data["t"]=[]
data["R"]=[]
data["M"]=[]
data['omega']=[]
data["I"]=[]
#data["lambda"]=[]
data["rho_c"]=[]
data["alpha"]=[]
data["beta"]=[]
data["xi1"]=[]

def Y(x):
    return 1-X(x)-Z
def mu(x,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    
    return 1/(5/4*X(x,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)+3/4-Z_/4)
def int_mu(x,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    return x**2*mu(x,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)
#mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))[0]
#print(mu_ser)
def F(x,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    return mu(x,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)/mu_ser

#dF=F(x_,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4).diff()
#dF=sm.lambdify(x_,dF)
#F(x_).diff()
def error(xi1,g,eps,mu_ser,*coef):
    temp_xi1=xi1
    xi,y=calc(temp_xi1,g,mu_ser,*coef)
    err=xi[-1]-temp_xi1
    while err>eps:
        temp_xi1+=err
        xi,y=calc(temp_xi1,g,mu_ser,*coef)
        err=xi[-1]-temp_xi1
        #print(err)
       #input()
    return xi,y

def g1(xi,y,u,xi1,mu_ser,coef):
    try:
        return -2/xi*u-y**3#*F(xi/xi1)**2#-(F1(0)/xi1)*dF(xi/xi1)
    except ZeroDivisionError:
        return -y**3
def g2(xi,y,u,xi1,mu_ser,coef):
    try:
        return -2/xi*u-y**3*F(xi/xi1,mu_ser,*coef)**2#-(F1(0)/xi1)*dF(xi/xi1)
    except ZeroDivisionError:
        return -y**3
def alpha_y(x,y,xi1,coef):
    return x**2*(y**3)*(mu(x/xi1,*coef)/mu(0,*coef))
def integ_y(xi,y,xi1,coef):
    yy=np.zeros(len(xi))
    for i in range(len(xi)-1):
        try:
            yy[i+1]=1/xi[i+1]**2*sp.integrate.quad(alpha_y,0,xi[i+1],args=(y[i+1],xi1,coef))[0]
        except ZeroDivisionError:
            pass
    return yy
def pade_mu(x):
    a0=0.0149173
    a1=-0.0868327
    a2=0.730856
    a3=1.7342
    b0=0.0172646
    b1=-0.0893741
    b2=1.0339
    b3=2.96529
    return(a0+a1*x+a2*x**2+a3*x**3)/(b0+b1*x+b2*x**2+b3*x**3)
def X_now(x):
    a0=0.000923382
    a1=-0.0036938
    a2=0.177201
    a3=-0.69752
    a4=3.856
    b0=0.00281341
    b1=-0.0112101
    b2=0.248745
    b3=-0.961501
    b4=5.43149
    return ((a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x)/(b0+b1*x+b2*x*x+b3*x*x*x+b4*x*x*x*x))
#mu0_ser=0.61328

def pade_f(x):
    mu0_ser=0.61328
    return pade_mu(x)/mu0_ser
pade_df=sm.lambdify(x_,pade_f(x_).diff())

y_t0 = pd.read_csv('y_res.txt', sep='\t+', header=None,engine='python')
y_t0.columns = ["xi", "y", "error"]

y_now = pd.read_csv('res_st.txt', sep='\t+', header=None,engine='python')
y_now.columns = ["xi", "y", "error"]

apy_t0=np.poly1d(np.polyfit(y_t0['xi'],y_t0['y'],14))

def alpa(x):
    return x**2*(apy_t0(x)**3)*(pade_mu(x/xi1)/pade_mu(0))
yy=[.359,.584,.648,.679,.694,.702,.705,.707,.708,.708,.708]

a0=0.00121312#0.000923382
a1=-0.0129071#-0.0036938
a2=0.159383#0.177201
a3=-0.899003#-0.69752
a4=2.39018#3.856
b0=0.00239208#0.00281341
b1=-0.0253873#-0.0112101
b2=0.252826#0.248745
b3=-1.31403#-0.961501
b4=3.40033#5.43149



xx=np.linspace(0,1,len(yy))
apr_x=np.poly1d(np.polyfit(xx,yy,9))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([ dcc.Graph(id="graph1")])
                ])
            ],width={"size":8}),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
            dbc.Row([
            dbc.Col([dbc.Label("a0"),
                     dbc.Input(type="number",min=-10,max=10,step=0.00001,id="a0",value=a0),]),
            dbc.Col([dbc.Label("b0"),
                     dbc.Input(type="number",min=-10,max=10,step=0.00001,id="b0",value=b0),])
                ]),
            dbc.Row([
            dbc.Col([dbc.Label("a1"),
                     dbc.Input(type="number",min=-10,max=10,step=0.00001,id="a1",value=a1),]),
            dbc.Col([dbc.Label("b1"),
                     dbc.Input(type="number",min=-10,max=10,step=0.00001,id="b1",value=b1),])
                ]),

            dbc.Row([
            dbc.Col([dbc.Label("a2"),
                     dbc.Input(type="number",min=-10,max=10,step=0.001,id="a2",value=a2),]),
            dbc.Col([dbc.Label("b2"),
                     dbc.Input(type="number",min=-10,max=10,step=0.001,id="b2",value=b2)])
                ]),
             dbc.Row([
            dbc.Col([dbc.Label("a3"),
                     dbc.Input(type="number",min=-10,max=10,step=0.001,id="a3",value=a3),]),
            dbc.Col([dbc.Label("b3"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="b3",value=b3),])
                ]),

            dbc.Row([
            dbc.Col([dbc.Label("a4"),
                     dbc.Input(type="number",min=-10,max=10,step=0.001,id="a4",value=a4),]),
            dbc.Col([dbc.Label("b4"),
                     dbc.Input(type="number",min=-10,max=10,step=0.001,id="b4",value=b4),])
                ]),
            html.Br(),
            dbc.Row([
                dbc.Col([dbc.Button("Calculate",  id="calc", n_clicks=0)],width={"size":4}),
                    dbc.Col([dbc.Button("save",  id="save",size="md", color="success",n_clicks=0)],width={"size":4}),
                    
                ])
            ,                       ]))
           
           
           

            ],width={"size":4})
       
]),
dbc.Row([
    dbc.Col([dbc.Spinner([ dcc.Graph(id="graph2")],size="sm")],width={"size":6}),
    dbc.Col([dbc.Spinner([dcc.Graph(id="graph3")],size="sm")],width={"size":6}),

    ]),
dbc.Row([
    dbc.Col([ 
             dbc.Card(
                 dbc.CardBody([
             dbc.Table(id="table")]))

        ],width={"size":12})
    ])
    ])
xi1_p=0
@app.callback(
            Output("a2","value"),
            [Input("save","n_clicks"),
             State("graph1","figure"),
             State("graph2","figure"),
             State("graph3","figure")]
        )

def save(n,x,y,rho):
    if n==0:
        return dash.no_update
    global data

    #print(data)
    

    data1={}
    data2={}
    data3={}
    x_x=x["data"][1]["x"]
    x_y=x["data"][1]["y"]

    xi_y=y["data"][1]["x"]
    y_y=y["data"][1]["y"]

    xi_rho=rho["data"][1]["x"]
    y_rho=rho["data"][1]["y"]

    data3["x"]=x_x

    data3["X"]=x_y

    data1["x"]=xi_rho
    data1["rho"]=y_rho

    data2["xi"]=xi_y
    data2["y"]=y_y

    
    df = pd.DataFrame(data=data)
    df.to_excel("parameters.xlsx")
    df1 = pd.DataFrame(data=data2)
    df1.to_excel("y(xi).xlsx")
    df2 = pd.DataFrame(data=data1)
    df2.to_excel("rho(x).xlsx")
    df3 = pd.DataFrame(data=data3)
    df3.to_excel("X(x).xlsx")

    data["t"]=[]
    data["R"]=[]
    data["M"]=[]
    data['omega']=[]
    data["I"]=[]
    #data["lambda"]=[]
    data["rho_c"]=[]
    data["alpha"]=[]
    data["xi1"]=[]
    data['beta']=[]

    


    return dash.no_update



def now_X(x):
    a0=0.000923382
    a1=-0.0036938
    a2=0.177201
    a3=-0.69752
    a4=3.856
    b0=0.00281341
    b1=-0.0112101
    b2=0.248745
    b3=-0.961501
    b4=5.43149   
    return ((a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x)/(b0+b1*x+b2*x*x+b3*x*x*x+b4*x*x*x*x))
 
@app.callback(
    Output('graph1', "figure"),
    [Input("a0", "value"),
    Input("a1", "value"),
    Input("a2", "value"),
    Input("a3", "value"),
    Input("a4", "value"),
    Input("b0", "value"),
    Input("b1", "value"),
    Input("b2", "value"),
     Input("b3","value"),
    Input("b4", "value")
     ]
)
def update_figure(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    #yy=[.359,.584,.648,.679,.694,.702,.705,.707,.708,.708,.708]
    #xx=np.linspace(0,1,len(yy))
    apr_x=np.poly1d(np.polyfit(xx,yy,9))
    #popt, pcov = sp.optimize.curve_fit(X, xx, yy)
    
    #print(popt[])
    #input()
    


    fig=go.Figure()
    #fig1=go.Figure()
    coef=a0,a1,a2,a3,a4,b0,b1,b2,b3,b4
    mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))[0]
    #print(mu_ser)
    #input()
    x=np.linspace(0,1,100)
    #mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a,b))[0]
    fig.add_trace(go.Scatter(
    x=x,
    y=now_X(x),
    name="X_now(x)"
    ))
    fig.add_trace(go.Scatter(
    x=x,
    y=X(x,*coef),
    name="X_future(x)"
    ))
    fig.add_trace(go.Scatter(
    x=x,
    y=[0.708 for _ in x],
    name="X_0(x)"
    ))

    fig.add_trace(go.Scatter(
    x=x,
    y=mu(x,*coef),
    name="mu(x)"
    ))
    fig.add_trace(go.Scatter(
    x=x,
    y=F(x,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4),
    name="f(x)"
    ))
    

    fig.update_layout(
        #legend_title_text='a2='+str(a)+" a4="+str(b),
        showlegend=True,
         xaxis=dict(
            tickmode = 'array',
            tickvals = np.linspace(0, 1, 5),
            ticktext = ['0', "0.25", '0.5', '0.75', '1']
         ),
        width=750,
        height=500,
        
        xaxis_title="x",
        yaxis_title="",
        
    ) 

    return fig
    
 #   return fig
@app.callback(
    Output("graph2","figure"),
    [State("graph1", "figure"),
     Input("calc","n_clicks"),
     State("a0","value"),
    State("a1","value"),
    State("a2","value"),
    State("a3","value"),
    State("a4","value"),
    State("b0","value"),
    State("b1","value"),
    State("b2","value"),
    State("b3","value"),
    State("b4","value"),
     ]
)
def update_figure1(fig,n,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
    if n==0:
        return dash.no_update
    else:
        fig1=go.Figure()
        coef=a0,a1,a2,a3,a4,b0,b1,b2,b3,b4
        mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))[0]
        dF=F(x_,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4).diff()
        dF=sm.lambdify(x_,dF)
        def iter_method():
            def p0(xi,y,u,xi1,mu_ser,coef):
                try:
                    return -2/xi*u-y**3*F(xi/xi1,mu_ser,*coef)**2-(F(0,mu_ser,*coef)/xi1)*dF(xi/xi1)
                except ZeroDivisionError:
                    return -y**3
            def p_i(xi,y,u,xi1,mu_ser,coef):
                try:
                    return -2/xi*u-y**3*F(xi/xi1,mu_ser,*coef)**2-(F(0,mu_ser,*coef)/xi1)*dF(xi/xi1)*apr_i(xi)
                except ZeroDivisionError:
                    return -y**3
            xi,y=calc(6.897,p0,mu_ser,*coef)

            while True:
                ##calc integral###
                xi1=xi[-1]
                i=integ_y(xi,y,xi[-1],coef)
                apr_i=np.poly1d(np.polyfit(xi,i,14))

                ##calc new y with integral###
                xi,y=calc(xi1,p_i,mu_ser,*coef)
                ## check if close enought###
                if np.sqrt((xi[-1] - xi1) ** 2) < 1e-3:
                    return xi,y

    
        xi,y=iter_method()
        fig1.add_trace(go.Scatter(
        x=np.array(y_t0["xi"]),#/7.72,
        y=y_t0["y"],
        name="y_now(xi)"
        ))
        fig1.add_trace(go.Scatter(
        x=xi,#/xi_me3[-1],
        y=y,
        name="y_future(xi)"
        ))
        fig1.add_trace(go.Scatter(
        x=np.array(y_now["xi"]),#/6.896,
        y=y_now["y"],
        name="y_0(xi)"
        ))

        fig1.update_layout(
            #legend_title_text='a2='+str(a)+" a4="+str(b),
            showlegend=True,
             xaxis=dict(
                tickmode = 'array',
                tickvals = np.linspace(0, xi[-1], 5),
                #ticktext = ['0', "0.25", '0.5', '0.75', '1']
             ),
            width=650,
            height=600,
            
            xaxis_title='xi',
            yaxis_title="",
            
        ) 

        return fig1



@app.callback(
    [Output("graph3","figure"),
     Output("table","children")],
    [Input("graph2", "figure"),
     State("a0","value"),
    State("a1","value"),
    State("a2","value"),
    State("a3","value"),
    State("a4","value"),
    State("b0","value"),
    State("b1","value"),
    State("b2","value"),
    State("b3","value"),
    State("b4","value"),
     ]
)
def update_figure2(fig,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4):
      
    R0=6.9634*10**10
    M0=1.9891*10**30
    R0_0=6.646*10**10
    L0=3.846*10**26
    mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))[0]


    rho_c=164.9420
    fig2=go.Figure()
    xi=fig["data"][1]["x"]
    #xi=[x*xi1_p for x in xi]
    y=fig["data"][1]["y"]
    xi1_=xi[-1]
    xi1=7.724
    #print(xi1_)
    apy=np.poly1d(np.polyfit(xi,y,14))
    print(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)
    def alpha_(x):
        return x**2*(apy(x)**3)*(mu(x/xi1_,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)/mu(0,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))


    a_=sp.integrate.quad(alpha_,0,xi1_)[0]



    


    a=1.30993
   # print("a_",a_)
    #print("a",a)
    #print(xi)
    rho_c_=((M0*xi1_**3)*(4*np.pi*a_*R0**3)**-1)*1000
    #print("rho_c",rho_c_)
    rho_c_now=88.50
    apr_y_now=np.poly1d(np.polyfit(y_now["xi"],y_now["y"],14))

    def rho_now(x):
        return rho_c_now*(apr_y_now(x))**3

    def rho(x):
        return rho_c*(pade_f(x/xi1)/pade_f(0))*apy_t0(x)**3

    def rho_(x):
        return rho_c_*(F(x/xi1_,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)/F(0,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))*apy(x)**3

    def I_(x):
        return x**4*(apy(x)**3)*(mu(x/xi1_,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)/mu(0,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))
    def I(x):
        return x**4*(apy_t0(x)**3)*(pade_mu(x/xi1)/pade_mu(0))
    
    def II(x):
        return x**4*rho_(x)
    def I_now(x):
        return x**4*rho_now(x)

    i_now=sp.integrate.quad(I_now,0,1)[0]
    #print(i_now)
    #input()
    i_=sp.integrate.quad(I_,0,xi1_)[0]
    i=sp.integrate.quad(I,0,xi1)[0]


    #print("-----")
    #print(i)
    #print(i_)
    #ii=sp.integrate.quad(II,0,1)[0]
    #print(ii)
    #input()
    #print(i_)
    #input()
  #  print(i_)
  #  print(i)


    omega=(a_/a)*(i/i_)*((xi1_**2)/(xi1**2))
   # print(omega)
   # input()
    def M_h0(x):
        return x**2*(apy_t0(x)**3)*(pade_mu(x/xi1)/pade_mu(0))*X_now(x/xi1)
    def M_h1(x):
        return x**2*(apy(x)**3)*(mu(x/xi1_,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)/mu(0,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))*X(x/xi1_,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)
    beta0=sp.integrate.quad(M_h0,0,xi1)[0]
    beta1=sp.integrate.quad(M_h1,0,xi1_)[0]
    

    #print(beta0)
    #print(beta1)

    #print("t=",((beta0/a)-(beta1/a_))*10.478*10)

    


    fig2.add_trace(go.Scatter(
    x=np.array(y_t0["xi"])/xi1,
    y=rho(y_t0["xi"]),
    name="rho_now(x)"
    ))

    fig2.add_trace(go.Scatter(
    x=np.array(xi)/xi1_,
    y=rho_(np.array(xi)),
    name="rho_future(x)"
    ))
    fig2.add_trace(go.Scatter(
    x=np.array(y_now["xi"])/6.896,
    y=rho_now(np.array(y_now["xi"])),
    name="rho_0(x)"
    ))

    
    

    fig2.update_layout(
        #legend_title_text='a2='+str(a)+" a4="+str(b),
        showlegend=True,
         xaxis=dict(
            tickmode = 'array',
            tickvals = np.linspace(0, 1, 5),
            #ticktext = ['0', "0.25", '0.5', '0.75', '1']
         ),
        width=650,
        height=600,
        
        xaxis_title="x",
        yaxis_title="",
        
    )
    #print(beta0,beta1)
    #t=((beta0/a)-(beta1/a_))*10.478*10+4.5
    t=(((2*M0)/(25.4*L0))*(0.708-(beta1/a_))*0.00716*(sp.constants.c**2))/31556952/10**6


    #gamma=(1/(4.5**2))*((R0/R0_0)-1)
    #R1=R0_0*(1+gamma*(t**2))
    k0=0.9545
    k1=0.0101
    R1=R0_0*(k0+(k1*t))

    ##masas
    masa=4*np.pi*a_*rho_c_*(R1/xi1_)**3

    
    global data
    
    data["t"].append(t)
    data["R"].append(R1/10**10)
    data["M"].append(masa)
    data['omega'].append((a_/a)*(i/i_)*((xi1_**2)/(xi1**2)))
    data["I"].append(i_)
    #data["lambda"].append(R1*(1/xi1_)/10**10)
    data["rho_c"].append(rho_c_)
    data["alpha"].append(a_)
    data["beta"].append(beta1)
    #data["mu_ser"]
    print(beta1,mu_ser)
    data["xi1"].append(xi1_)
        
    

    df1=pd.DataFrame(data=data)
    table=dbc.Table.from_dataframe(df1,bordered=True)
    return fig2,[table]

import webbrowser
if __name__=="__main__":
    #webbrowser.open("http://127.0.0.1:8050/")

    app.run_server(debug=True)

    #webbrowser.open("http://127.0.0.1:8050/")
    #main()
