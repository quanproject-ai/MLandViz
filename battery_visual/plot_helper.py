import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

COLORS = px.colors.qualitative.Plotly

def _generate_subplot(rows:int,cols:int,legend_title:str,x_title:str,y_title:str,**kwargs):
    fig = make_subplots(rows=rows,cols=cols)
    fig.update_layout(legend_title_text =legend_title)
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return fig

def _boxplot(fig,y,x,color,row,col,*args):
    fig.add_trace(go.Box(
        y=y,
        name=x,
        showlegend=True,
        marker_color=color
    ),
    row=row,
    col=col
)

def _scatter(fig,y,x,colors,name,row=None,col=None,legend = None,*args, **kwargs):
    fig.add_trace(go.Scatter(
        name = name,
        y=y,
        mode='markers',
        x=x,
        marker_color = colors,
        showlegend= legend
    ),
    row=row,
        col=col)