FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set the non-root username and group
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV SHELL=/bin/bash
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get install -y bash
RUN ln -sf /bin/bash /bin/sh
RUN chown -R vscode /opt/conda
CMD ["bash"]
EXPOSE 3000 8080 8081 8082

USER $USERNAME
RUN conda init
RUN conda update conda -y
RUN conda install -y  geopandas==1.0.1 opencv==4.10.0 rasterio==1.4.1 && conda clean -afy
RUN pip install --no-cache-dir lightning==2.4.0



