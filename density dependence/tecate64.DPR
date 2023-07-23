library tecate64;
{$R+,I+,Q+}

// Density dependence for Ceanothus verrucosus
// changed so that program only exits if K<0 not <=0
// Plants compete for resources and die or progress to next age class.
// Amount of space used ("SpaceUsed") is determined by plant size ("AvgWeight");
// it is the sum (over all age classes) of the product of plant size and number of plants in the stage.
// occurrence of veg repro will be the only time there will be more than one age class presence in a stand.

// If SpaceUsed > CarryingCapacity, survival and growth are reduced as follows.

// Survival is the total survival from an age class. Under density dependent competition it is reduced by
// (1-Mortality).  Mortality is specific to a "age group" (set of
// similar ages, such as all juvenile ages).  Mortality is specified
// in RAMAS Metapop, under Model|Populations|User-defined.
// The age groups (i.e., the beginning and ending stage for each "group") are
// defined in this code.  Changing these definitions would require re-compiling
// the DLL.

// Growth is the proportion of survivors moving to the next stage (=Subdiagonal/Survival)
// Since CEVE is an age based model all survivors move to the next stage.
// We are leaving ChangeinGrowht in this model but setting it to 0 so it has no effect.
// Changing these definitions would require re-compiling the DLL.

// This program requires that the model has the following:
// * 50 stages, grouped into "stage groups" as defined in this code
// * Survival transitions are defined by the diagonal and subdiagonal elements
//   of the stage matrix (except for the last stage, which has only diagonal).
// * Average plant size is entered as "Average weight" and calculated based on the N plants/m2
// * Total available space is entered as "Carrying capacity (K)"

{$undef test}

uses
  DLL_Intf;

const
  UserParameterCount = 6;
  UserParameterNames: array[1..UserParameterCount] of PChar = (
    'Age 1-2 Mort',  // Mortality due to shading in each stage group
    'Age 3-4 Mort',
    'Age 5-6 Mort',
    'Age 7-8 Mort',
    'Age 9-18 Mort',
    'Age 19-100+ Mort');

  StageGroups = UserParameterCount div 1;

  //The following are the age class  numbers for stage group
First : array[1..StageGroups] of integer = (1,2,3,4,5,10);
  Last   : array[1..StageGroups] of integer = (1,2,3,4,9,50);

  UserParameterMinimums: array[1..UserParameterCount] of Double = (
    0, 0, 0, 0, 0, 0);
  UserParameterMaximums: array[1..UserParameterCount] of Double = (
    1, 1, 1, 1, 1, 1);
  UserParameterDefaults: array[1..UserParameterCount] of Double = (
    0.6, 0.5, 0.4, 0.3, 0.2, 0.1); 
  UserParameterDecimals: array[1..UserParameterCount] of Cardinal = (
    4, 4, 4, 4, 4, 4);

//---------------------------------------------------------------------------
procedure GetUserParametersInfo(
  var names   : TPCharArrayPtr;
  var minimums: TDoubleArrayPtr;
  var maximums: TDoubleArrayPtr;
  var defaults: TDoubleArrayPtr;
  var decimals: TCardinalArrayPtr;
  var count   : Integer); stdcall;
begin
  names    := @UserParameterNames;
  minimums := @UserParameterMinimums;
  maximums := @UserParameterMaximums;
  defaults := @UserParameterDefaults;
  decimals := @UserParameterDecimals;
  count    := High(UserParameterNames);
end; //GetUserParametersInfo


procedure InitializeSimulation(
      Replications  : Cardinal;   //number of replications for simulation
      Duration      : Cardinal;   //number of timsteps for simulation
      Populations   : Cardinal;   //number of populations
      Stages        : Cardinal;   //number of stages
  var UserErrorCode : Integer); stdcall;
begin
//popcount:=populations;
end; //InitializeSimulation

procedure FinalizeSimulation; stdcall;
begin
end; //FinalizeSimulation


//---------------------------------------------------------------------------
procedure ComputeDensityDependence(
//This DLL should NOT change the values of the following parameters:
      ThisReplication   : Cardinal; //current replication of the simulation
      Replications      : Cardinal; //number of replications in this simulation
      Timestep          : Cardinal; //current time step of the simulation
      Duration          : Cardinal; //duration (number of time steps) in this simulation
      PopIndex          : Cardinal; //index for this population
      Stages            : Cardinal; //number of stages in the model

      SexStructure      : SexStructureType; //which sex(es) modeled; see manual

      // The following 3 parameters should be used ONLY if SexStructure=TwoSexes
      FemaleStages      : Cardinal;         //number of female stages
      MatingSystem      : MatingSystemType; //see manual
      AverageMates      : Double;           //females per male (if polygynous) or
                                          //males per female (if polyandrous)
                                          //=1 (if monogamous)

      // stage arrays (vectors of stage-specific parameters)
      AvgWeight         : TSingleArrayPtr;  //average weight of individuals in stage
      Exclude           : TBooleanArrayPtr; //whether stage is excluded from total N
      BasisForDD        : TBooleanArrayPtr; //whether stage is used as a basis for
                                          //dens dep (used only if DDbasis=selectedstages)
      Reproductive      : TDoubleArrayPtr;  //proportion of reproductive individuals

      //DD parameters
      DDaffects         : DDactionType;     //which vital rates affected by DD
      DDbasis           : DDBasisType;      //which abundance is DD based on
      Rmax              : Double;           //maximum growth rate for this population
      CarryingCapacity  : Double;           //K for this population

      //Abundance parameters
      N                 : Integer;          //total abundance of this population
      Nstage            : TLongIntArrayPtr; //abundance in each stage
      //Note: N = sum of Nstage (even if exclude=True for some stages)

      ConstraintsMatrix : TDoubleMatrixPtr; //see manual

      UserParams        : TDoubleArrayPtr;  //user-defined parameters

      //This DLL can change the following parameters
      TMatrix           : TDoubleMatrixPtr; //Transition matrix for this population
      //Incorporates "Relative survival" and "Relative fecundity" parameters

      //Error handling:
  var userDDErrorCode     : Integer;      //code for error in this function (see below)
  var userDDErrorMsg      : TCharArray);  // message string about the error (see below)
      //code=0 : no error
      //code=1 : display the message at the end of simulation
      //code>1 : terminate simulation immediately and display the message
      //code<0 : display the message immediately, continue simulation, set code=0,
      //         and delete the message (thus, 1 message displayed per function call)

  stdcall; // THIS LINE MUST REMAIN

var
  i,s : Integer;
  j : Cardinal;
  extra: double;
  DDmortality:double;
  Mortality : array [1..StageGroups] of double;
  Survival, Growth, SpaceUsed : double;

Procedure CheckInput;
  begin
  if stages<>50 then
      begin
      userDDErrorCode := 2;
      userDDErrorMsg  := 'This function requires a model with 50 stages';
      end;

  end; //CheckInput

var fo:text;

begin //Procedure ComputeDensityDependence

// if thisreplication=0 then exit;

  Mortality[1] := userParams[1];
  Mortality[2] := userParams[2];
  Mortality[3] := userParams[3];
  Mortality[4] := userParams[4];
  Mortality[5] := userParams[5];
  Mortality[6] := userParams[6];
  

// Check consistency of input
if (thisreplication<=1) and (timestep=1) then CheckInput;

if CarryingCapacity<0 then exit;

// Density dependence: plants are shaded by those in the same stage or in later
// (larger) stages.  Space used is determined by plant size ('AvgWeight')

SpaceUsed:= 0;

for s:=StageGroups downto 1 do
  begin

{$ifdef test}
writeln(fo);
writeln(fo,'Stage group ',s,': ');
flush(fo);
{$endif}

    // calculate spaceused by individuals in this stage group; add to the
    // spaceused by the later (larger) stages
    for j:=First[s] to Last[s] do if (Nstage[j]>0) and (AvgWeight^[j]>0) then
       SpaceUsed:= SpaceUsed + Nstage^[j] * AvgWeight^[j];

    // check if space is full (i.e., if plants are shaded)
    if (SpaceUsed > CarryingCapacity) then
       begin

       // reduce survival and growth for stages in this stage group
       for j:=First[s] to Last[s] do
          begin // for each stage in this stage group:
           if j=Stages then
             begin // for the final stage, there is only diagonal element
             Tmatrix^[j]^[j] := (1-Mortality[s]) *  Tmatrix^[j]^[j];
             end
           else // for all other stages:
             begin

             // Survival is (diagonal + subdiagonal)
             Survival := Tmatrix^[j]^[j]+Tmatrix^[j+1]^[j];

             // check
             if (survival<=0) then
                begin
                userDDErrorCode := 2;
                userDDErrorMsg  := 'Total survival = 0!';
                exit;
                end; // check

             // Growth is proportion of survivors that move to next stage
             Growth   := Tmatrix^[j+1]^[j] / Survival;

             //check
             if Growth>1 then Growth:=1;

             {Modify survival and growth when N is above K}
             Survival := (1-Mortality[s]) *  Survival;
           
             {Modify diagonal and subdiagonal matrix elements under shade}
             Tmatrix^[j+1]^[j] := Growth * Survival;
             Tmatrix^[j]^[j]   := Survival - Tmatrix^[j+1]^[j];

             //check
             if Tmatrix^[j]^[j]<0 then Tmatrix^[j]^[j]:=0;

             end; // else; other stages
           end; // j; stages


       end; // if (Spaceused>K)
  end; // s= StageGroups downto 1

{$ifdef test}
close(fo);
{$endif}

end; // Procedure ComputeDensityDependence


exports
  InitializeSimulation,
  FinalizeSimulation,
  ComputeDensityDependence,
  GetUserParametersInfo;

// the following code lets the compiler verify that the function definitions
// match the type definitions
var
  ComputeDensityDependenceFunctionCheck : ComputeDensityDependenceFunctionType;
  GetUserParametersInfoFunctionCheck    : GetUserParametersInfoFunctionType;

begin
  ComputeDensityDependenceFunctionCheck := ComputeDensityDependence;
  if Assigned(ComputeDensityDependenceFunctionCheck) then
    ;

  GetUserParametersInfoFunctionCheck := GetUserParametersInfo;
  if Assigned(GetUserParametersInfoFunctionCheck) then
    ;
end.