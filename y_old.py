from jupyter_dash import JupyterDash

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
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

def calc(xi1,g,mu_ser,*coef):
    def rk(xi,y,u,h,g,xi1,*coef):
        k1=h*f(xi,y,u)
        l1=h*g(xi,y,u,xi1,mu_ser,*coef)
        k2=h*f(xi+h/2,y+k1/2,u+l1/2)
        l2=h*g(xi+h/2,y+k1/2,u+l1/2,xi1,mu_ser,*coef)
        k3=h*f(xi+h/2,y+k2/2,u+l2/2)
        l3=h*g(xi+h/2,y+k2/2,u+l2/2,xi1,mu_ser,*coef)
        k4=h*f(xi+h,y+k3,u+l3)
        l4=h*g(xi+h,y+k3,u+l3,xi1,mu_ser,*coef)
        return y0+1/6*(k1+2*k2+2*k3+k4),u0+1/6*(l1+2*l2+2*l3+l4)
    h=1e-3
    y0=1
    u0=0
    xi0=0
    y=[y0]
    du=[u0]
    xi=[xi0]
    try:
        while y[-1]>0:
            y0,u0=rk(xi0,y0,u0,h,g,xi1,coef)
            xi0+=h
            y.append(y0)
            xi.append(xi0)
            du.append(u0)
        return xi,y
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
        

def Y(x):
    return 1-X(x)-Z_
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
    xi,y=calc(temp_xi1,g,mu_ser,coef)
    err=xi[-1]-temp_xi1
    while err>eps:
        temp_xi1+=err
        xi,y=calc(temp_xi1,g,mu_ser,coef)
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
        return -2/xi*u-y**3*F(xi/xi1,mu_ser,coef)**2#-(F1(0)/xi1)*dF(xi/xi1)
    except ZeroDivisionError:
        return -y**3
def alpha_y(x,y,xi1):
    return x**2*(y**3)*(mu(x/xi1)/mu(0))
def integ_y(xi,y,xi1):
    yy=[]
    for i in range(len(xi)):
        try:
            yy.append(1/xi[i]**2*sp.integrate.quad(alpha_y,0,xi[i],args=(y[i],xi1))[0])
        except ZeroDivisionError:
            yy.append(0)
    return yy
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="graph1"),
            dcc.Graph(id="graph2")

            ],
                width={"size":8}),
        dbc.Col([
            dbc.Row([
            dbc.Col([dbc.Label("a0"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="a0",value=4.50096e-05),]),
            dbc.Col([dbc.Label("b0"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="b0",value=0.000448987),])
                ]),
            dbc.Row([
            dbc.Col([dbc.Label("a1"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="a1",value=-0.000669583),]),
            dbc.Col([dbc.Label("b1"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="b1",value=-0.00399509),])
                ]),

            dbc.Row([
            dbc.Col([dbc.Label("a2"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="a2",value=0.0358607),]),
            dbc.Col([dbc.Label("b2"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="b2",value=0.0745353)])
                ]),
             dbc.Row([
            dbc.Col([dbc.Label("a3"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="a3",value=-0.53596),]),
            dbc.Col([dbc.Label("b3"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="b3",value=-0.827868),])
                ]),

            dbc.Row([
            dbc.Col([dbc.Label("a4"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="a4",value=4.97482),]),
            dbc.Col([dbc.Label("b4"),
                     dbc.Input(type="number",min=-10,max=10,step=0.01,id="b4",value=7.08559),])
                ]),


           
           
           

            ],width={"size":4})
       
])
    ])
@app.callback(
    [Output('graph1', 'figure'),
     Output("graph2","figure")],
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
    fig=go.Figure()
    fig1=go.Figure()
    coef=a0,a1,a2,a3,a4,b0,b1,b2,b3,b4
    mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a0,a1,a2,a3,a4,b0,b1,b2,b3,b4))[0]

    x=np.linspace(0,1,100)
    #mu_ser=3*sp.integrate.quad(int_mu,0,1,args=(a,b))[0]
    fig.add_trace(go.Scatter(
    x=x,
    y=X(x,*coef),
    name="X(x)"
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
        width=600,
        height=600,
        
        xaxis_title="x",
        yaxis_title="",
        
    )
    """
    dF=F(x_,mu_ser,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4).diff()
    dF=sm.lambdify(x_,dF)
    xi_me,y_me=calc(6.897,g1,mu_ser,coef)
    xi_me2,y_me2=error(6.897,g2,1e-4,mu_ser,coef)
    apy=np.poly1d(np.polyfit(xi_me,y_me,14))
    apy2=np.poly1d(np.polyfit(xi_me2,y_me2,14))
    i_apr=np.poly1d(np.polyfit(xi_me2,i_ser,14))
    def g3(xi,y,u,xi1,):
        try:
            return -2/xi*u-y**3*F(xi/xi1,mu_ser,*coef)**2-(F(0,mu_ser,*coef)/xi1)*dF(xi/xi1,mu_ser,*coef)*i_apr(xi)
        except ZeroDivisionError:
            return -y**3
    xi_me3,y_me3=error(xi_me2[-1],g3,1e-2,mu_ser,coef)
    #print(xi_me3[-1])
    #i_me3=integ(xi_me3,y_me3,10.5)
    fig1.add_trace(go.Scatter(
    x=xi_me3,
    y=y_me3,
    name="y(x)"
    ))
    """
    fig1.update_layout(
        #legend_title_text='a2='+str(a)+" a4="+str(b),
        showlegend=True,
         xaxis=dict(
            tickmode = 'array',
            tickvals = np.linspace(0, 1, 5),
            #ticktext = ['0', "0.25", '0.5', '0.75', '1']
         ),
        width=600,
        height=600,
        
        xaxis_title="x",
        yaxis_title="",
        
    )



    return fig,fig1
import webbrowser
if __name__=="__main__":
    webbrowser.open("http://127.0.0.1:8050/")

    app.run_server(debug=True)

    #webbrowser.open("http://127.0.0.1:8050/")
    #main()
